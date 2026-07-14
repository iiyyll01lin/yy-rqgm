/**
 * Deterministic VRAM / bandwidth physics for the MOCK fallback.
 *
 * Kept intentionally consistent with the backend "Static Hardware Gatekeeper"
 * (`backend/gatekeeper/vram.py` + `bandwidth.py`) so the mock produces the same
 * first-principles numbers the live backend would — the UI stays credible with
 * or without a server:
 *
 *   M_weights   = N_params × bytes_per_param
 *   KV_per_tok  = 2 × n_layers × n_kv_heads × head_dim × 2   (KV kept at fp16)
 *   M_kv        = KV_per_tok × (prefix + population × branch) (prefix-cache aware)
 *   M_act       = 0.10 × (weights + KV)                       (activation fraction)
 *   VRAM_total  = weights + KV + activations + overhead(1.0 GB fixed)
 *   tokens/s    ≈ 0.7 × mem_bandwidth / bytes_read_per_token  (decode is mem-bound)
 *
 * All memory values are decimal GB (bytes / 1e9).
 */

import { DTYPE_BYTES, type Dtype, type VramBreakdown } from "./types";

/** Fixed framework / ROCm context + allocator reservation (matches backend). */
const OVERHEAD_GB = 1.0;
/** Transient activation working set as a fraction of (weights + KV). */
const ACTIVATION_FRACTION = 0.1;
/** Achievable fraction of peak memory bandwidth during decode. */
const BANDWIDTH_EFFICIENCY = 0.7;
/** KV cache is kept at fp16 (2 B/element) regardless of weight quantization. */
const KV_BYTES = 2;

const clamp = (v: number, lo: number, hi: number) =>
  Math.min(hi, Math.max(lo, v));

const round = (v: number, dp = 2) => {
  const f = 10 ** dp;
  return Math.round(v * f) / f;
};

export interface VramInput {
  paramsB: number;
  nLayers: number;
  nKvHeads: number;
  headDim: number;
  hidden: number;
  seqLen: number;
  /** Parallel sequences / batch / evolutionary population. */
  population: number;
  dtype: string;
  /** Fraction (0..1) of each sequence shared as a cached prefix. */
  prefixRatio?: number;
}

/** Bytes-per-parameter for the weights, defaulting to fp16 for unknowns. */
export function weightBytes(dtype: string): number {
  return DTYPE_BYTES[dtype as Dtype] ?? 2;
}

/** KV bytes stored per token across all layers (both K and V), fp16. */
export function kvPerTokenBytes(
  nLayers: number,
  nKvHeads: number,
  headDim: number,
): number {
  return 2 * nLayers * nKvHeads * headDim * KV_BYTES;
}

/**
 * Effective KV token count after prefix caching. A shared prefix of ratio r is
 * materialized once; the remaining (1 - r) is paid per sequence.
 */
function effectiveKvTokens(
  seqLen: number,
  population: number,
  prefixRatio: number,
): number {
  const r = clamp(prefixRatio, 0, 1);
  const prefixLen = Math.floor(seqLen * r);
  const branchLen = seqLen - prefixLen;
  return prefixLen + population * branchLen;
}

/** Full VRAM breakdown for a given workload. */
export function computeVram(input: VramInput): {
  breakdown: VramBreakdown;
  total: number;
} {
  const {
    paramsB,
    nLayers,
    nKvHeads,
    headDim,
    seqLen,
    population,
    dtype,
    prefixRatio = 0,
  } = input;

  const weights = paramsB * weightBytes(dtype);
  const kvPerTok = kvPerTokenBytes(nLayers, nKvHeads, headDim);
  const kv =
    (kvPerTok * effectiveKvTokens(seqLen, population, prefixRatio)) / 1e9;
  const activations = ACTIVATION_FRACTION * (weights + kv);
  const overhead = OVERHEAD_GB;

  const breakdown: VramBreakdown = {
    weights: round(weights),
    kv_cache: round(kv),
    activations: round(activations),
    overhead: round(overhead),
  };
  const total = round(
    breakdown.weights +
      breakdown.kv_cache +
      breakdown.activations +
      breakdown.overhead,
    1,
  );
  return { breakdown, total };
}

/**
 * Memory-bandwidth bound decode throughput estimate. Per decode step the engine
 * streams all weights plus the current sequence's KV cache from memory.
 */
export function tokensPerSecond(
  bandwidthTbs: number,
  input: Pick<
    VramInput,
    "paramsB" | "nLayers" | "nKvHeads" | "headDim" | "seqLen" | "dtype"
  >,
): number {
  const weightsBytes = input.paramsB * 1e9 * weightBytes(input.dtype);
  const kvReadBytes =
    kvPerTokenBytes(input.nLayers, input.nKvHeads, input.headDim) *
    input.seqLen;
  const bytesPerToken = weightsBytes + kvReadBytes;
  if (bytesPerToken <= 0) return 0;
  const bwBytes = bandwidthTbs * 1e12;
  return Math.round((BANDWIDTH_EFFICIENCY * bwBytes) / bytesPerToken);
}

/**
 * Maximum number of parallel sequences (population) a memory budget can hold,
 * accounting for a once-materialized shared prefix. Mirrors the backend's
 * `feasibility.max_population` (overhead + (weights + shared_prefix)·(1+act)
 * fixed, each branch costs branch_KV·(1+act)).
 */
export function maxPopulation(memoryGb: number, input: VramInput): number {
  const r = clamp(input.prefixRatio ?? 0, 0, 1);
  const weights = input.paramsB * weightBytes(input.dtype);
  const perTokGb =
    kvPerTokenBytes(input.nLayers, input.nKvHeads, input.headDim) / 1e9;
  const prefixLen = Math.floor(input.seqLen * r);
  const branchLen = input.seqLen - prefixLen;
  const sharedPrefixGb = perTokGb * prefixLen;
  const perBranchGb = perTokGb * branchLen;

  const act = 1 + ACTIVATION_FRACTION;
  const fixed = OVERHEAD_GB + (weights + sharedPrefixGb) * act;
  const perBranchEff = perBranchGb * act;

  const available = memoryGb - fixed;
  if (available < 0) return 0;
  if (perBranchEff <= 0) return 1_000_000;
  return Math.max(0, Math.floor(available / perBranchEff));
}

/**
 * KV savings (%) from prefix caching versus the naive per-sequence baseline.
 * Equals r × (P - 1) / P (matches the backend's prefix_savings_pct).
 */
export function prefixSavingsPct(
  population: number,
  prefixRatio: number,
): number {
  const r = clamp(prefixRatio, 0, 1);
  if (population <= 1) return 0;
  return round(r * ((population - 1) / population) * 100, 1);
}
