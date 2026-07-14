"""Admin endpoints: RQGM epoch upgrade (propose challenger / HITL-gated approve)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from backend.app.models import (
    EpochApproveRequest,
    EpochApproveResponse,
    EpochProposeResponse,
)
from backend.app.sessions import get_store
from backend.evaluator import evolve

router = APIRouter(prefix="/api/admin", tags=["admin"])

# The latest proposed challenger awaiting an approve/reject decision (PoC state).
_pending_challenger: str | None = None


@router.post("/epoch/propose", response_model=EpochProposeResponse)
def propose_epoch() -> EpochProposeResponse:
    """Reflectively mutate the champion rubric into a scored challenger."""
    global _pending_challenger
    store = get_store()
    feedback = evolve.load_seed_feedback() + store.feedback_log
    proposal = evolve.propose_challenger(feedback=feedback, traces=store.trace_log)
    _pending_challenger = proposal.version
    return EpochProposeResponse(
        challenger_id=proposal.version,
        rubric_diff=proposal.rubric_diff,
        metrics=proposal.metrics,
    )


@router.post("/epoch/approve", response_model=EpochApproveResponse)
def approve_epoch(body: EpochApproveRequest) -> EpochApproveResponse:
    """HITL gate: promote the pending challenger (epoch++) or reject it."""
    global _pending_challenger
    if _pending_challenger is None:
        raise HTTPException(
            status_code=400,
            detail="no challenger proposed; POST /api/admin/epoch/propose first",
        )
    result = evolve.approve_challenger(_pending_challenger, approve=body.approve)
    if result.get("applied"):
        _pending_challenger = None  # consumed
    return EpochApproveResponse(epoch_id=result["epoch_id"], applied=result["applied"])
