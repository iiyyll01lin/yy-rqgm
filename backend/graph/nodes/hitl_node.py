"""HITL node: pause for human review via LangGraph ``interrupt()``.

The graph is compiled with a SqliteSaver checkpointer, so this interrupt is
durable: the run pauses, state is persisted under the session's thread_id, and
the API resumes it with ``Command(resume=decision)``. Human feedback here is the
ground-truth anchor that gates RQGM epoch upgrades (prevents reward hacking).
"""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from backend.graph.state import GraphState


def hitl_node(state: GraphState) -> dict[str, Any]:
    request = {
        "type": "approval_request",
        "prompt": "Approve this proposed architecture? Provide {approved: bool, notes?: str}.",
        "architecture": state.get("architecture", ""),
        "deficit_score": state.get("deficit_score"),
        "red_flags": state.get("red_flags", []),
        "gate_report": state.get("gate_report", {}),
        "epoch_id": state.get("epoch_id", 0),
    }

    # Pauses here on first pass; resumes with the value from Command(resume=...).
    decision = interrupt(request)

    if isinstance(decision, dict):
        approved = bool(decision.get("approved", False))
        decision_payload = decision
    else:
        approved = bool(decision)
        decision_payload = {"approved": approved}

    trace = list(state.get("trace", []))
    trace.append(f"hitl: decision approved={approved}")

    return {
        "awaiting_hitl": False,
        "hitl_decision": decision_payload,
        "approved": approved,
        "trace": trace,
    }
