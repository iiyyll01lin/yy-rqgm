"""Typed State schema for the orchestration graph.

An explicit, typed State (rather than ad-hoc dict coupling) is itself one of the
things the RQGM evaluator rewards — so the orchestrator practices what it judges.
"""

from __future__ import annotations

from typing import Any, TypedDict


class GraphState(TypedDict, total=False):
    # --- inputs -----------------------------------------------------------
    session_id: str
    domain: str
    need: str
    model_id: str
    tier_id: str
    seq_len: int
    concurrency: int
    dtype: str
    prefix_ratio: float

    # --- router -----------------------------------------------------------
    matched_template_id: str
    matched_templates: list[dict[str, Any]]

    # --- task agent -------------------------------------------------------
    architecture: str
    proposal: dict[str, Any]
    agent_program_version: str

    # --- gatekeeper (DETERMINISTIC hard gate) -----------------------------
    feasible: bool
    gate_report: dict[str, Any]
    gate_rejection: str | None

    # --- rqgm evaluator (FUZZY) ------------------------------------------
    deficit_score: float
    red_flags: list[dict[str, Any]]
    reasoning: str
    epoch_id: int

    # --- HITL -------------------------------------------------------------
    awaiting_hitl: bool
    hitl_request: dict[str, Any]
    hitl_decision: dict[str, Any]
    approved: bool

    # --- diagnostics ------------------------------------------------------
    trace: list[str]
    used_mock: bool
