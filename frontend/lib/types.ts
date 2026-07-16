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

/* --------------------------------------------------------------------- */
/* Admin / RQGM epoch-evolution contract                                  */
/*                                                                        */
/* Shapes for the epoch-admin surface: the two-stage promotion gate       */
/* (code gate first, HITL veto-only), the Pareto frontier population       */
/* search, RQGM hack-ratio / exploitation state, and the transparency      */
/* report (data splits, val/test separation, judge agreement, memory).     */
/* --------------------------------------------------------------------- */

/** Per-split anchor counts (train / dev / val / test isolation). */
export interface SplitCount {
  weak: number;
  strong: number;
  total: number;
}

/**
 * Anchor dataset split counts across the four disjoint splits (data isolation):
 * `train` drives GEPA mutation, `dev` RANKS/SELECTS the Pareto frontier, `val`
 * is scored ONLY by the code gate, and `test` is reporting-only. Selecting on
 * `dev` (a split the gate never sees) removes the winner's curse so the `val`
 * gate is a genuine held-out re-test.
 */
export interface DataSplits {
  train: SplitCount;
  dev: SplitCount;
  val: SplitCount;
  test: SplitCount;
}

/**
 * Reproducibility provenance recorded with a report: which judge model + source
 * revision produced the numbers. NOTE: this block is emitted by the backend's
 * `build_report`, but is optional here because the `EpochReportResponse` pydantic
 * model does not (yet) declare it, so it is only guaranteed in Mock mode.
 */
export interface Provenance {
  judge_model: string;
  using_mock: boolean | null;
  rqgm_backend: string;
  git_sha: string | null;
}

/**
 * Over-optimization monitor: proxy(val) − gold(test) separation gap. A large
 * positive gap means the champion looks much sharper on the split closer to the
 * optimization loop (`val`, the gate split) than on the untouched gold `test`.
 */
export interface OverOptimization {
  proxy_val_separation: number;
  gold_test_separation: number;
  separation_gap: number;
}

/** One adversarial sample in the over-acceptance monitor. */
export interface OverAcceptanceSample {
  id: string;
  targets: string[];
  deficit: number;
  accepted_as_strong: boolean;
  out_of_catalog: boolean;
}

/**
 * Over-acceptance monitor: fraction of gamed/weak adversarial samples (incl.
 * out-of-catalog gaming) the current rubric wrongly scores as "strong"
 * (deficit < tau). High = the judge is fooled by disguised poison pills.
 */
export interface OverAcceptance {
  over_acceptance_rate: number;
  accepted_as_strong: number;
  n: number;
  tau: number;
  per_sample: OverAcceptanceSample[];
}

/** Weak-vs-strong deficit separation on one held-out split. */
export interface SplitSeparation {
  separation: number;
  mean_weak_deficit: number;
  mean_strong_deficit: number;
  n: number;
}

/**
 * RQGM hack-ratio / exploitation assessment (+ automatic tolerance tightening).
 * `mean_hack_ratio` is `mean(quality_strict / quality_loose)`; low values mean
 * the rubric passes (loose) what the ground-truth poison-pill check fails
 * (strict), so exploitation fires and the tolerance schedule is tightened.
 */
export interface ExploitationReport {
  backend: string;
  mean_hack_ratio: number | null;
  exploitation_detected: boolean;
  reason: string;
  tolerances_before: number[];
  tolerances_after: number[];
  tightened: boolean;
  trigger_adversarial_injection: boolean;
  strictness_level: number;
  n_samples: number;
}

/** One anchor's judge-vs-human prediction (for the agreement confusion view). */
export interface AgreementAnchor {
  id: string;
  deficit: number;
  predicted: "weak" | "strong";
  label: "weak" | "strong";
}

/** Judge/human agreement on one split (accuracy + Cohen's κ). */
export interface JudgeAgreement {
  n: number;
  tau: number;
  accuracy: number;
  cohen_kappa: number;
  correct: number;
  per_anchor: AgreementAnchor[];
}

/** A single Pareto-frontier member (a candidate challenger rubric). */
export interface FrontierMember {
  version: string;
  objectives: Record<string, number>;
  bbe: number;
  added_criteria: string[];
  parent_version: string;
  // Thompson-sampling bandit state: how many children this member parented and
  // how many of them were gate-improving (survived Pareto + raised BBε).
  child_successes?: number;
  child_trials?: number;
}

/** Pareto frontier summary — top-K non-dominated challenger rubrics. */
export interface Frontier {
  size: number;
  top_k: number;
  objectives: string[];
  best_version: string | null;
  members: FrontierMember[];
}

/** Qdrant (or local fallback) memory statistics. */
export interface MemoryStats {
  mode: string;
  collection?: string;
  total: number;
  heuristic_failure: number;
  physics_truth: number;
  error?: string;
}

/** Response for GET /api/admin/report — the RQGM transparency report. */
export interface EpochReport {
  epoch_id: number;
  champion_version: string;
  rqgm_backend: string;
  // Optional: emitted by build_report but not declared on the backend response
  // model, so only guaranteed present in Mock mode (rendered when available).
  provenance?: Provenance;
  data_splits: DataSplits;
  separation: { val: SplitSeparation; test: SplitSeparation };
  // proxy(val) − gold(test) over-optimization gap.
  over_optimization: OverOptimization;
  // fraction of gamed/weak adversarial samples wrongly scored "strong".
  over_acceptance: OverAcceptance;
  hack_ratio: ExploitationReport;
  judge_agreement: { val: JudgeAgreement; test: JudgeAgreement };
  frontier: Frontier;
  memory: MemoryStats;
}

/** Per-flaw-family win/loss/tie breakdown for the P2 posterior (transparency). */
export interface FlawWinCounts {
  win: number;
  loss: number;
  tie: number;
}

/**
 * The code gate result (Stage 1) — the anti-reward-hacking core, scored ONLY on
 * the held-out `val` split:
 * - **P1 non-inferiority**: `challenger_sep > champion_sep` (ties favour the incumbent);
 * - **P2 Bayesian Beta-Binomial posterior**: `posterior_prob_improvement`
 *   (= `P(Δsep>0)` from a Beta posterior over per-anchor paired win indicators on
 *   the weak anchors) must clear `posterior_threshold`, AND the observed
 *   `effect_size` (= `separation_delta`) must clear `min_detectable_effect` (MDE).
 * A per-flaw-family win/loss/tie breakdown is reported for transparency.
 */
export interface GateResult {
  passed: boolean;
  reason: string;
  champion_separation: number;
  challenger_separation: number;
  separation_delta: number;
  p1_non_inferior: boolean;
  p2_passed: boolean;
  // P2 (Bayesian) transparency — replaces the old bootstrap CI fields.
  posterior_prob_improvement: number; // P(Δsep>0) = P(θ>0.5)
  posterior_threshold: number;
  effect_size: number; // == separation_delta (the MDE test target)
  min_detectable_effect: number; // MDE
  n_wins: number;
  n_losses: number;
  n_ties: number;
  n_val: number;
  n_weak: number;
  per_flaw_wins: Record<string, FlawWinCounts>;
  champion_deficits: Record<string, number>;
  challenger_deficits: Record<string, number>;
}

/**
 * HITL result (Stage 2) — a veto-only safety lock. The human is `consulted`
 * ONLY when the code gate passes; they can `veto` a passing challenger but can
 * NEVER override a failed code gate (`consulted:false` when the gate fails).
 */
export interface HitlResult {
  consulted: boolean;
  approved: boolean | null;
  vetoed: boolean;
}

/**
 * Metrics for the frontier's best member returned by /epoch/propose. Separation
 * is reported on the `dev` SELECTION split (`split: "dev"`); the code gate
 * independently re-checks on the held-out `val` split. (The single-mutation
 * fallback reports `split: "train"`.)
 */
export interface ProposeMetrics {
  split: string;
  frontier_size?: number;
  added_criteria?: string[];
  champion_separation: number;
  challenger_separation: number;
  separation_delta: number;
  bbe_lower_bound?: number;
  objectives?: Record<string, number>;
  note?: string;
}

/** Response for POST /api/admin/epoch/propose. */
export interface EpochProposeResponse {
  challenger_id: string;
  rubric_diff: string;
  metrics: ProposeMetrics;
  frontier: Frontier;
}

/** Request body for POST /api/admin/epoch/approve. */
export interface EpochApproveRequest {
  approve: boolean;
}

/** Response for POST /api/admin/epoch/approve (the two-stage gate outcome). */
export interface EpochApproveResponse {
  epoch_id: number;
  applied: boolean;
  champion_version: string;
  gate: GateResult;
  hitl: HitlResult;
  champion_exploitation: ExploitationReport;
  challenger_exploitation: ExploitationReport;
  erased_memories: number;
  reconfirmed_memories: number;
  reason: string;
}

/** Compact evaluator summary added to GET /health. */
export interface EvaluatorHealth {
  rqgm_backend: string;
  val_separation: number;
  test_separation: number;
  // proxy(val) − gold(test) over-optimization gap.
  proxy_gold_separation_gap: number;
  // fraction of gamed/weak adversarial samples wrongly scored "strong".
  over_acceptance_rate: number;
  hack_ratio: number | null;
  exploitation_detected: boolean;
  tolerance_levels: number;
  val_judge_accuracy: number;
  val_judge_kappa: number;
}

/** Inference backend status surfaced on GET /health. */
export interface InferenceHealth {
  using_mock: boolean;
  base_url: string;
}

/** Subset of GET /health consumed by the epoch-admin surface. */
export interface HealthResponse {
  status: string;
  epoch_id: number;
  champion_version: string;
  inference?: InferenceHealth;
  memory?: MemoryStats;
  evaluator?: EvaluatorHealth | { error: string };
}
