"""Admin endpoints: RQGM epoch upgrade (propose challenger / HITL-gated approve)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import (
    EpochApproveRequest,
    EpochApproveResponse,
    EpochProposeResponse,
    EpochReportResponse,
)
from backend.app.sessions import get_store
from backend.evaluator import evolve, report

router = APIRouter(prefix="/api/admin", tags=["admin"])

# The latest proposed challenger awaiting an approve/reject decision (PoC state).
_pending_challenger: str | None = None


@router.post("/epoch/propose", response_model=EpochProposeResponse)
def propose_epoch() -> EpochProposeResponse:
    """Propose a challenger via GEPA Pareto-frontier population search.

    Runs a red-team pass to generate poison-pill samples targeting the current
    champion, evolves a Pareto frontier of rubrics against them, and returns the
    frontier's best member (by anchor BBε) for the code gate to judge.
    """
    global _pending_challenger
    store = get_store()
    feedback = evolve.load_seed_feedback() + store.feedback_log
    adversarial = evolve.generate_adversarial_pool()
    proposal, frontier = evolve.propose_via_frontier(
        feedback=feedback, traces=store.trace_log, adversarial_samples=adversarial
    )
    _pending_challenger = proposal.version
    return EpochProposeResponse(
        challenger_id=proposal.version,
        rubric_diff=proposal.rubric_diff,
        metrics=proposal.metrics,
        frontier=frontier.to_dict(),
    )


@router.post("/epoch/approve", response_model=EpochApproveResponse)
def approve_epoch(body: EpochApproveRequest) -> EpochApproveResponse:
    """Two-stage epoch upgrade.

    Stage 1 is a CODE gate (held-out anchor separation: P1 non-inferiority + P2
    Bayesian Beta-Binomial posterior over paired win indicators, gated on a
    minimum practical effect). Stage 2 is the HITL boolean, which acts ONLY as a
    final safety veto *after* the code gate passes — it can reject a passing
    challenger but can never override a failed gate.
    """
    global _pending_challenger
    if _pending_challenger is None:
        raise HTTPException(
            status_code=400,
            detail="no challenger proposed; POST /api/admin/epoch/propose first",
        )
    result = evolve.approve_challenger(_pending_challenger, approve=body.approve)
    if result.get("applied"):
        _pending_challenger = None  # consumed
    return EpochApproveResponse(
        epoch_id=result["epoch_id"],
        applied=result["applied"],
        champion_version=result.get("champion_version", ""),
        gate=result.get("gate", {}),
        hitl=result.get("hitl", {}),
        champion_exploitation=result.get("champion_exploitation", {}),
        challenger_exploitation=result.get("challenger_exploitation", {}),
        erased_memories=result.get("erased_memories", 0),
        reconfirmed_memories=result.get("reconfirmed_memories", 0),
        reason=result.get("reason", ""),
    )


@router.get("/report", response_model=EpochReportResponse)
def epoch_report() -> EpochReportResponse:
    """RQGM transparency report: val/test separation, hack ratio (strict/loose),
    judge/human agreement (accuracy + Cohen's κ), the latest Pareto frontier, and
    memory stats — all on held-out anchors under the current champion."""
    data = report.build_report()
    return EpochReportResponse(**data)
