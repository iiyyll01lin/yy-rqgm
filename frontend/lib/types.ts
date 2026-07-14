/**
 * TypeScript definitions for the AgentForge backend REST contract.
 *
 * These shapes mirror the API described in the platform blueprint and are the
 * single source of truth shared by the live API client and the mock fallback
 * layer (so the UI stays fully demoable even with no backend running).
 */

/** Quantization / compute precisions surfaced in the wizard controls. */
export type Dtype = "int4" | "fp8" | "fp16" | "bf16";

/** Ordered precision options used by the UI selectors. */
export const DTYPES: Dtype[] = ["int4", "fp8", "fp16"];

/** Approximate bytes-per-parameter for each precision (weights). */
export const DTYPE_BYTES: Record<Dtype, number> = {
  int4: 0.5,
  fp8: 1,
  fp16: 2,
  bf16: 2,
};

/** Broad AMD hardware families, from edge NPU up to data-center Instinct. */
export type TierClass = "Ryzen AI" | "Radeon" | "Radeon PRO" | "Instinct";

/** A single AMD hardware tier returned by GET /api/tiers. */
export interface Tier {
  id: string;
  name: string;
  class: TierClass;
  memory_gb: number;
  bandwidth_tbs: number;
  form_factor: string;
  has_npu: boolean;
  tops_npu?: number;
  price_usd_est?: number;
  notes: string;
}

/** A model spec returned by GET /api/models. */
export interface ModelSpec {
  id: string;
  name: string;
  params_b: number;
  n_layers: number;
  n_kv_heads: number;
  head_dim: number;
  hidden: number;
  context_len: number;
  dtype_default: string;
}

/** Response for POST /api/session. */
export interface SessionResponse {
  session_id: string;
}

/** A domain workflow template matched by the router. */
export interface WorkflowTemplate {
  id: string;
  name: string;
  description: string;
  needs: string[];
}

/** Request body for POST /api/session/{id}/domain. */
export interface DomainRequest {
  domain: string;
  description: string;
  workload_type?: string;
}

/** Response for POST /api/session/{id}/domain. */
export interface DomainResponse {
  matched_templates: WorkflowTemplate[];
  // Backend may return null when no template matches; the wizard tolerates it.
  recommended_template_id: string | null;
}

/** Deterministic VRAM breakdown (all values in GiB). */
export interface VramBreakdown {
  weights: number;
  kv_cache: number;
  activations: number;
  overhead: number;
}

/** Ordered keys + human labels for rendering the stacked VRAM bar. */
export const VRAM_SEGMENTS: {
  key: keyof VramBreakdown;
  label: string;
  hint: string;
}[] = [
  { key: "weights", label: "模型權重 Weights", hint: "N_params × bytes/param" },
  {
    key: "kv_cache",
    label: "KV 快取 KV Cache",
    hint: "2 × layers × kv_heads × head_dim × seq × batch",
  },
  {
    key: "activations",
    label: "啟動值 Activations",
    hint: "前向傳遞的暫存張量 working set",
  },
  {
    key: "overhead",
    label: "框架開銷 Overhead",
    hint: "ROCm/框架 context、allocator、CUDA graph",
  },
];

/** Current hardware selection for the diagnostic step. */
export interface CurrentHardware {
  tier_id?: string;
  custom?: {
    memory_gb: number;
    bandwidth_tbs: number;
  };
}

/** Workload requirements for the diagnostic step. */
export interface Requirements {
  model_id: string;
  seq_len: number;
  concurrency: number;
  dtype: string;
}

/** Request body for POST /api/session/{id}/diagnose. */
export interface DiagnoseRequest {
  current_hardware: CurrentHardware;
  requirements: Requirements;
}

/** The deterministic gatekeeper report. */
export interface DiagnoseReport {
  vram_total_gb: number;
  vram_breakdown: VramBreakdown;
  tokens_per_s_est: number;
  headroom_gb: number;
}

/** A single capability gap explained in the customer's domain language. */
export interface Gap {
  constraint: string;
  needed: string;
  have: string;
  explanation_domain: string;
}

/** Response for POST /api/session/{id}/diagnose. */
export interface DiagnoseResponse {
  feasible: boolean;
  report: DiagnoseReport;
  gaps: Gap[];
}

/** Request body for POST /api/session/{id}/simulate. */
export interface SimulateRequest {
  model_id: string;
  seq_len: number;
  population: number;
  dtype: string;
  prefix_ratio?: number;
  tier_ids?: string[];
}

/** Per-tier simulation result (the Hardware Simulation Lab centerpiece). */
export interface TierSimResult {
  tier_id: string;
  feasible: boolean;
  vram_total_gb: number;
  vram_breakdown: VramBreakdown;
  tokens_per_s_est: number;
  max_population: number;
  kv_savings_from_prefix_pct: number;
}

/** Response for POST /api/session/{id}/simulate. */
export interface SimulateResponse {
  per_tier: TierSimResult[];
}

/** Request body for POST /api/session/{id}/export. */
export interface ExportRequest {
  target_tier_id: string;
  model_id: string;
  template_id: string;
  // Optional sizing knobs so the exported TCO reflects the simulated scenario
  // (the backend defaults these when omitted).
  seq_len?: number;
  concurrency?: number;
  dtype?: string;
  prefix_ratio?: number;
}

/** Response for POST /api/session/{id}/export. */
export interface ExportResponse {
  tco_markdown: string;
  deploy_files: Record<string, string>;
}

/** Request body for POST /api/session/{id}/feedback. */
export interface FeedbackRequest {
  rating: number;
  correct?: boolean;
  notes: string;
}

/** Response for POST /api/session/{id}/feedback. */
export interface FeedbackResponse {
  ok: boolean;
  stored_as: string;
}
