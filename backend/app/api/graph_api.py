"""Supplementary endpoints exposing the LangGraph orchestrator (b4).

These are NOT part of the fixed REST contract but make the compiled StateGraph
(Task Agent -> Gatekeeper -> RQGM Evaluator -> HITL interrupt) drivable/testable
over HTTP, with one durable thread per session.
"""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, HTTPException

from backend.app.models import OrchestrateRequest, OrchestrateResumeRequest
from backend.app.sessions import get_store
from backend.graph.orchestrator import resume_run, start_run

router = APIRouter(prefix="/api", tags=["graph"])

_MERMAID = (
    "flowchart TD\n"
    "  START --> router\n"
    "  router --> task_agent\n"
    "  task_agent --> gatekeeper\n"
    "  gatekeeper -->|feasible| rqgm_evaluator\n"
    "  gatekeeper -->|infeasible: HARD REJECT| END\n"
    "  rqgm_evaluator --> hitl\n"
    "  hitl -->|interrupt for approval| END\n"
)


@router.get("/graph")
def describe_graph() -> dict[str, Any]:
    return {
        "nodes": ["router", "task_agent", "gatekeeper", "rqgm_evaluator", "hitl"],
        "edges": [
            ["START", "router"],
            ["router", "task_agent"],
            ["task_agent", "gatekeeper"],
            ["gatekeeper", "rqgm_evaluator (if feasible)"],
            ["gatekeeper", "END (hard reject if infeasible)"],
            ["rqgm_evaluator", "hitl"],
            ["hitl", "END (interrupt)"],
        ],
        "mermaid": _MERMAID,
        "notes": "gatekeeper is deterministic and never evolves; rqgm_evaluator is frozen within an epoch.",
    }


def _session_or_404(sid: str) -> dict:
    sess = get_store().get(sid)
    if sess is None:
        raise HTTPException(status_code=404, detail=f"unknown session: {sid}")
    return sess


@router.post("/session/{sid}/orchestrate")
def orchestrate(sid: str, body: OrchestrateRequest) -> dict[str, Any]:
    sess = _session_or_404(sid)
    inputs: dict[str, Any] = {
        "session_id": sid,
        "domain": sess.get("domain"),
        "need": body.need or sess.get("description") or sess.get("domain") or "",
        "seq_len": body.seq_len,
        "concurrency": body.concurrency,
        "prefix_ratio": body.prefix_ratio,
        "trace": [],
    }
    if body.model_id:
        inputs["model_id"] = body.model_id
    if body.tier_id:
        inputs["tier_id"] = body.tier_id
    if body.dtype:
        inputs["dtype"] = body.dtype
    return start_run(inputs, thread_id=sid)


@router.post("/session/{sid}/orchestrate/resume")
def orchestrate_resume(sid: str, body: OrchestrateResumeRequest) -> dict[str, Any]:
    _session_or_404(sid)
    return resume_run({"approved": body.approved, "notes": body.notes}, thread_id=sid)
