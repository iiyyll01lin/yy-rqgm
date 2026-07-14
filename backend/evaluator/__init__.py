"""RQGM Evaluator — the FUZZY, EVOLVING half of the platform.

Physically separate from ``backend.gatekeeper`` on purpose: this package holds
the domain judgement that *evolves* (GEPA-style reflective mutation, RQGM
epoch/HITL gating, selective erasure). It must never make physical-feasibility
decisions — that is the deterministic gatekeeper's job.
"""

from backend.evaluator.judge import (  # noqa: F401
    Evaluation,
    RedFlag,
    evaluate_architecture,
)

__all__ = ["Evaluation", "RedFlag", "evaluate_architecture"]
