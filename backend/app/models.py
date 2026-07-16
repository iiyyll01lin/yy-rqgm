"""Pydantic v2 request/response models mirroring the REST API contract exactly."""

from __future__ import annotations

from pydantic import BaseModel, Field

# ---------------------------------------------------------------------------
# session
# ---------------------------------------------------------------------------
class SessionResponse(BaseModel):
    session_id: str


# ---------------------------------------------------------------------------
# catalog: tiers / models
# ---------------------------------------------------------------------------
class TierOut(BaseModel):
    id: str
    name: str
    # 'class' is a reserved word; expose it via alias so JSON key is "class".
    tier_class: str = Field(alias="class")
    memory_gb: float
    bandwidth_tbs: float
    form_factor: str
    has_npu: bool
    tops_npu: float | None = None
    price_usd_est: float | None = None
    notes: str = ""

    model_config = {"populate_by_name": True}


class TiersResponse(BaseModel):
    tiers: list[TierOut]


class ModelOut(BaseModel):
    id: str
    name: str
    params_b: float
    n_layers: int
    n_kv_heads: int
    head_dim: int
    hidden: int
    context_len: int
    dtype_default: str


class ModelsResponse(BaseModel):
    models: list[ModelOut]


# ---------------------------------------------------------------------------
# domain
# ---------------------------------------------------------------------------
class DomainRequest(BaseModel):
    domain: str
    description: str
    workload_type: str | None = None


class MatchedTemplate(BaseModel):
    id: str
    name: str
    description: str
    needs: list[str] = Field(default_factory=list)


class DomainResponse(BaseModel):
    matched_templates: list[MatchedTemplate]
    recommended_template_id: str | None = None


# ---------------------------------------------------------------------------
# diagnose
# ---------------------------------------------------------------------------
class CustomHardware(BaseModel):
    memory_gb: float
    bandwidth_tbs: float


class CurrentHardware(BaseModel):
    tier_id: str | None = None
    custom: CustomHardware | None = None


class Requirements(BaseModel):
    model_id: str
    seq_len: int = 4096
    concurrency: int = 1
    dtype: str | None = None


class DiagnoseRequest(BaseModel):
    current_hardware: CurrentHardware
    requirements: Requirements


class VramBreakdownModel(BaseModel):
    weights: float
    kv_cache: float
    activations: float
    overhead: float


class DiagnoseReport(BaseModel):
    vram_total_gb: float
    vram_breakdown: VramBreakdownModel
    tokens_per_s_est: float
    headroom_gb: float


class Gap(BaseModel):
    constraint: str
    needed: float
    have: float
    explanation_domain: str


class DiagnoseResponse(BaseModel):
    feasible: bool
    report: DiagnoseReport
    gaps: list[Gap] = Field(default_factory=list)


# ---------------------------------------------------------------------------
# simulate
# ---------------------------------------------------------------------------
class SimulateRequest(BaseModel):
    model_id: str
    seq_len: int = 4096
    population: int = 1
    dtype: str | None = None
    prefix_ratio: float | None = 0.0
    tier_ids: list[str] | None = None


class PerTierResult(BaseModel):
    tier_id: str
    feasible: bool
    vram_total_gb: float
    vram_breakdown: VramBreakdownModel
    tokens_per_s_est: float
    max_population: int
    kv_savings_from_prefix_pct: float


class SimulateResponse(BaseModel):
    per_tier: list[PerTierResult]


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------
class EvaluateRequest(BaseModel):
    architecture: str
    domain: str | None = None


class RedFlagModel(BaseModel):
    criterion: str
    severity: str
    detail: str


class EvaluateResponse(BaseModel):
    deficit_score: float
    red_flags: list[RedFlagModel] = Field(default_factory=list)
    reasoning: str = ""
    epoch_id: int = 0
    # Physics-surrogate verdict on the numerical-duct-tape failure mode (advisory).
    surrogate: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# export
# ---------------------------------------------------------------------------
class ExportRequest(BaseModel):
    target_tier_id: str
    model_id: str
    template_id: str | None = None
    # optional sizing knobs (sensible defaults keep the contract minimal)
    seq_len: int = 8192
    concurrency: int = 1
    dtype: str | None = None
    prefix_ratio: float = 0.0


class ExportResponse(BaseModel):
    tco_markdown: str
    deploy_files: dict[str, str]


# ---------------------------------------------------------------------------
# feedback
# ---------------------------------------------------------------------------
class FeedbackRequest(BaseModel):
    rating: int
    correct: bool | None = None
    notes: str = ""


class FeedbackResponse(BaseModel):
    ok: bool
    stored_as: str = "ground_truth_anchor"


# ---------------------------------------------------------------------------
# admin / epoch
# ---------------------------------------------------------------------------
class EpochProposeResponse(BaseModel):
    challenger_id: str
    rubric_diff: str
    metrics: dict
    frontier: dict = Field(default_factory=dict)


class EpochApproveRequest(BaseModel):
    approve: bool


class EpochApproveResponse(BaseModel):
    epoch_id: int
    applied: bool
    champion_version: str = ""
    # Two-stage gate transparency (needed by the epoch-admin UI + report):
    #   gate  = code-gate result (P1 non-inferiority + P2 Bayesian posterior/MDE);
    #   hitl  = whether the human was consulted / approved / vetoed;
    #   *_exploitation = RQGM hack-ratio + tolerance state.
    gate: dict = Field(default_factory=dict)
    hitl: dict = Field(default_factory=dict)
    champion_exploitation: dict = Field(default_factory=dict)
    challenger_exploitation: dict = Field(default_factory=dict)
    erased_memories: int = 0
    reconfirmed_memories: int = 0
    # Co-evolution coupling: on an evaluator promotion the AGENT archive is
    # re-scored under the new champion (agent-utility selective erasure).
    agent_utility_erasure: dict = Field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# admin / agent (RQGM agent-half co-evolution: propose / promote a program)
# ---------------------------------------------------------------------------
class AgentProposeResponse(BaseModel):
    # None when the frontier never beat the incumbent champion program.
    challenger_id: str | None = None
    metrics: dict = Field(default_factory=dict)
    frontier: dict = Field(default_factory=dict)


class AgentPromoteResponse(BaseModel):
    agent_epoch_id: int
    applied: bool
    champion_version: str = ""
    # Agent gate = challenger vs champion PROGRAM on held-out val needs, scored by
    # the epoch-FROZEN champion EVALUATOR (the RQGM asymmetry).
    gate: dict = Field(default_factory=dict)
    reason: str = ""


# ---------------------------------------------------------------------------
# admin / report (RQGM transparency: val/test separation, hack ratio, agreement)
# ---------------------------------------------------------------------------
class EpochReportResponse(BaseModel):
    epoch_id: int
    champion_version: str
    rqgm_backend: str
    # provenance = reproducibility metadata (judge_model, using_mock, rqgm_backend,
    # git_sha) so a report is tied to the exact model + source revision that made it.
    provenance: dict = Field(default_factory=dict)
    data_splits: dict = Field(default_factory=dict)
    separation: dict = Field(default_factory=dict)
    # over_optimization = proxy(val)-gold(test) separation gap;
    # over_acceptance   = fraction of gamed/weak adversarial samples scored strong.
    over_optimization: dict = Field(default_factory=dict)
    over_acceptance: dict = Field(default_factory=dict)
    hack_ratio: dict = Field(default_factory=dict)
    judge_agreement: dict = Field(default_factory=dict)
    # judge_vs_gold = agreement (accuracy + κ) with a STRONGER "gold" model as a
    # human-proxy labeler; empty unless a gold pass has run. NOT a real-human κ.
    judge_vs_gold: dict = Field(default_factory=dict)
    frontier: dict = Field(default_factory=dict)
    memory: dict = Field(default_factory=dict)
    # agent = the AGENT half's champion program + utility on held-out needs (scored
    # by the frozen champion evaluator); co_evolution = the two-halves epoch state +
    # agent-mined adversarial coupling. Both empty if the agent half is unavailable.
    agent: dict = Field(default_factory=dict)
    co_evolution: dict = Field(default_factory=dict)


# ---------------------------------------------------------------------------
# supplementary: LangGraph orchestration (exposes the b4 graph via API)
# ---------------------------------------------------------------------------
class OrchestrateRequest(BaseModel):
    need: str | None = None
    model_id: str | None = None
    tier_id: str | None = None
    seq_len: int = 4096
    concurrency: int = 1
    dtype: str | None = None
    prefix_ratio: float = 0.0


class OrchestrateResumeRequest(BaseModel):
    approved: bool
    notes: str = ""
