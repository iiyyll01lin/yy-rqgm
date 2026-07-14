"""Session-scoped wizard endpoints (the 4-step flow + evaluate/export/feedback)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import (
    DiagnoseReport,
    DiagnoseRequest,
    DiagnoseResponse,
    DomainRequest,
    DomainResponse,
    EvaluateRequest,
    EvaluateResponse,
    ExportRequest,
    ExportResponse,
    FeedbackRequest,
    FeedbackResponse,
    Gap,
    MatchedTemplate,
    PerTierResult,
    RedFlagModel,
    SessionResponse,
    SimulateRequest,
    SimulateResponse,
    VramBreakdownModel,
)
from backend.app.sessions import get_store
from backend.domains.base import WorkflowTemplate
from backend.domains.registry import get_domain, list_domains
from backend.evaluator import versioning
from backend.evaluator.judge import evaluate_architecture
from backend.export import build_deploy_files, generate_tco_markdown
from backend.gatekeeper import feasibility
from backend.gatekeeper.spec import get_model, get_tier, list_tiers
from backend.graph.router import route_need
from backend.memory import MemoryType, get_memory

router = APIRouter(prefix="/api", tags=["session"])

# throughput below this (tokens/s) triggers a soft "bandwidth" advisory gap
_SLOW_TPS = 20.0


def _session_or_404(sid: str) -> dict:
    store = get_store()
    sess = store.get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {sid}")
    return sess


def _find_template(template_id: str | None, domain_id: str | None) -> WorkflowTemplate | None:
    if not template_id:
        return None
    # ``domain_id`` may be a free-form industry label (not a registered pack id);
    # in that case search every domain so a selected template is still resolved.
    pack = get_domain(domain_id) if domain_id else None
    packs = [pack] if pack is not None else list_domains()
    for pack in packs:
        if pack is None:
            continue
        for tmpl in pack.workflow_templates():
            if tmpl.id == template_id:
                return tmpl
    return None


# ---------------------------------------------------------------------------
@router.post("/session", response_model=SessionResponse)
def create_session() -> SessionResponse:
    return SessionResponse(session_id=get_store().create())


# --- step 1: domain -------------------------------------------------------
@router.post("/session/{sid}/domain", response_model=DomainResponse)
def set_domain(sid: str, body: DomainRequest) -> DomainResponse:
    _session_or_404(sid)
    matches = route_need(body.description, domain_id=body.domain, top_k=5)
    matched = [
        MatchedTemplate(
            id=m.template.id,
            name=m.template.name,
            description=m.template.description,
            needs=m.template.needs,
        )
        for m in matches
    ]
    recommended = matches[0].template.id if matches else None
    get_store().update(
        sid,
        domain=body.domain,
        description=body.description,
        workload_type=body.workload_type,
        matched_templates=[m.model_dump() for m in matched],
        recommended_template_id=recommended,
    )
    return DomainResponse(matched_templates=matched, recommended_template_id=recommended)


# --- step 2: diagnose -----------------------------------------------------
@router.post("/session/{sid}/diagnose", response_model=DiagnoseResponse)
def diagnose(sid: str, body: DiagnoseRequest) -> DiagnoseResponse:
    sess = _session_or_404(sid)
    model = get_model(body.requirements.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {body.requirements.model_id}")

    seq_len = body.requirements.seq_len
    concurrency = body.requirements.concurrency
    dtype = body.requirements.dtype

    hw = body.current_hardware
    if hw.custom is not None:
        capacity_gb = hw.custom.memory_gb
        res = feasibility.evaluate_custom(
            memory_gb=hw.custom.memory_gb,
            bandwidth_tbs=hw.custom.bandwidth_tbs,
            model=model,
            seq_len=seq_len,
            concurrency=concurrency,
            dtype=dtype,
        )
        hw_name = f"custom ({hw.custom.memory_gb:.0f} GB)"
    elif hw.tier_id:
        tier = get_tier(hw.tier_id)
        if tier is None:
            raise HTTPException(status_code=404, detail=f"unknown tier: {hw.tier_id}")
        capacity_gb = tier.memory_gb
        res = feasibility.evaluate_tier(
            tier, model, seq_len=seq_len, concurrency=concurrency, dtype=dtype
        )
        hw_name = tier.name
    else:
        raise HTTPException(
            status_code=400, detail="current_hardware requires tier_id or custom{memory_gb,bandwidth_tbs}"
        )

    report = res.to_dict()
    domain = sess.get("domain") or "your deployment"

    gaps: list[Gap] = []
    if not res.feasible:
        gaps.append(
            Gap(
                constraint="vram_gb",
                needed=round(res.vram_total_gb, 2),
                have=round(capacity_gb, 2),
                explanation_domain=(
                    f"To run {model.name} for {domain} at {seq_len:,}-token context "
                    f"(x{concurrency}), you need ~{res.vram_total_gb:.0f} GB of VRAM, but "
                    f"{hw_name} has only {capacity_gb:.0f} GB. Move up to a higher-VRAM AMD "
                    f"tier (e.g. W7900 48 GB or MI300X 192 GB), quantize to int4, or reduce context."
                ),
            )
        )
    if res.tokens_per_s_est < _SLOW_TPS:
        gaps.append(
            Gap(
                constraint="throughput_tokens_per_s",
                needed=_SLOW_TPS,
                have=round(res.tokens_per_s_est, 1),
                explanation_domain=(
                    f"Decode is memory-bandwidth bound: ~{res.tokens_per_s_est:.0f} tokens/s on "
                    f"{hw_name}. For {domain} this may feel slow; a higher-bandwidth tier "
                    f"(HBM-class Instinct) would materially improve interactivity."
                ),
            )
        )

    return DiagnoseResponse(
        feasible=res.feasible,
        report=DiagnoseReport(
            vram_total_gb=report["vram_total_gb"],
            vram_breakdown=VramBreakdownModel(**report["vram_breakdown"]),
            tokens_per_s_est=report["tokens_per_s_est"],
            headroom_gb=report["headroom_gb"],
        ),
        gaps=gaps,
    )


# --- step 3: simulate -----------------------------------------------------
@router.post("/session/{sid}/simulate", response_model=SimulateResponse)
def simulate(sid: str, body: SimulateRequest) -> SimulateResponse:
    _session_or_404(sid)
    model = get_model(body.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {body.model_id}")

    prefix_ratio = body.prefix_ratio or 0.0
    if body.tier_ids:
        tiers = [get_tier(t) for t in body.tier_ids]
        tiers = [t for t in tiers if t is not None]
    else:
        tiers = list_tiers()

    per_tier: list[PerTierResult] = []
    for tier in tiers:
        res = feasibility.evaluate_tier(
            tier,
            model,
            seq_len=body.seq_len,
            concurrency=body.population,
            dtype=body.dtype,
            prefix_ratio=prefix_ratio,
        )
        rep = res.to_dict()
        per_tier.append(
            PerTierResult(
                tier_id=tier.id,
                feasible=res.feasible,
                vram_total_gb=rep["vram_total_gb"],
                vram_breakdown=VramBreakdownModel(**rep["vram_breakdown"]),
                tokens_per_s_est=rep["tokens_per_s_est"],
                max_population=rep["max_population"],
                kv_savings_from_prefix_pct=rep["kv_savings_from_prefix_pct"],
            )
        )
    return SimulateResponse(per_tier=per_tier)


# --- evaluate (RQGM judge) ------------------------------------------------
@router.post("/session/{sid}/evaluate", response_model=EvaluateResponse)
def evaluate(sid: str, body: EvaluateRequest) -> EvaluateResponse:
    sess = _session_or_404(sid)
    domain = body.domain or sess.get("domain")
    ev = evaluate_architecture(body.architecture, domain_id=domain)
    payload = {
        "deficit_score": ev.deficit_score,
        "red_flags": [rf.to_dict() for rf in ev.red_flags],
        "reasoning": ev.reasoning,
        "epoch_id": ev.epoch_id,
        "architecture": body.architecture,
        "domain": domain,
    }
    get_store().record_evaluation(sid, payload)
    return EvaluateResponse(
        deficit_score=ev.deficit_score,
        red_flags=[RedFlagModel(**rf.to_dict()) for rf in ev.red_flags],
        reasoning=ev.reasoning,
        epoch_id=ev.epoch_id,
    )


# --- step 4: export -------------------------------------------------------
@router.post("/session/{sid}/export", response_model=ExportResponse)
def export(sid: str, body: ExportRequest) -> ExportResponse:
    sess = _session_or_404(sid)
    model = get_model(body.model_id)
    if model is None:
        raise HTTPException(status_code=404, detail=f"unknown model: {body.model_id}")
    tier = get_tier(body.target_tier_id)
    if tier is None:
        raise HTTPException(status_code=404, detail=f"unknown tier: {body.target_tier_id}")

    domain = sess.get("domain")
    template = _find_template(body.template_id, domain)

    res = feasibility.evaluate_tier(
        tier,
        model,
        seq_len=body.seq_len,
        concurrency=body.concurrency,
        dtype=body.dtype,
        prefix_ratio=body.prefix_ratio,
    )
    tco = generate_tco_markdown(
        tier,
        model,
        res,
        template=template,
        domain_id=domain,
        seq_len=body.seq_len,
        concurrency=body.concurrency,
        dtype=body.dtype,
    )
    files = build_deploy_files(
        model,
        tier,
        res,
        template=template,
        domain_id=domain,
        max_len=body.seq_len,
        dtype=body.dtype,
    )
    return ExportResponse(tco_markdown=tco, deploy_files=files)


# --- feedback (ground-truth anchor) ---------------------------------------
@router.post("/session/{sid}/feedback", response_model=FeedbackResponse)
def feedback(sid: str, body: FeedbackRequest) -> FeedbackResponse:
    _session_or_404(sid)
    record = {
        "rating": body.rating,
        "correct": body.correct,
        "notes": body.notes,
        "epoch": versioning.get_epoch(),
        "session_id": sid,
    }
    get_store().record_feedback(sid, record)

    # A negative HITL result becomes a heuristic_failure memory tied to this epoch,
    # which selective erasure will prune when the epoch is superseded.
    if body.correct is False and body.notes:
        try:
            get_memory().add(
                text=body.notes,
                memory_type=MemoryType.HEURISTIC_FAILURE,
                created_at_epoch=versioning.get_epoch(),
                extra={"session_id": sid, "rating": body.rating},
            )
        except Exception:
            pass

    return FeedbackResponse(ok=True, stored_as="ground_truth_anchor")
