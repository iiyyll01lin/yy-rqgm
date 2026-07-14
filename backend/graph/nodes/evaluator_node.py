"""RQGM Evaluator node: FUZZY, evolving domain judgement (frozen this epoch)."""

from __future__ import annotations

from typing import Any

from backend.evaluator.judge import evaluate_architecture
from backend.graph.state import GraphState


def rqgm_evaluator_node(state: GraphState) -> dict[str, Any]:
    trace = list(state.get("trace", []))
    architecture = state.get("architecture", "")
    domain_id = state.get("domain")

    evaluation = evaluate_architecture(architecture, domain_id=domain_id)
    trace.append(
        f"rqgm_evaluator: deficit={evaluation.deficit_score:.3f}, "
        f"{len(evaluation.red_flags)} red flag(s), epoch={evaluation.epoch_id}"
    )

    return {
        "deficit_score": evaluation.deficit_score,
        "red_flags": [rf.to_dict() for rf in evaluation.red_flags],
        "reasoning": evaluation.reasoning,
        "epoch_id": evaluation.epoch_id,
        "trace": trace,
    }
