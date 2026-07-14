"""Gatekeeper node: DETERMINISTIC hard gate. Physics never evolves.

Calls the pure gatekeeper math. If the proposed workload is physically infeasible
on the chosen tier, it HARD-REJECTS — the fuzzy evaluator never even runs. This
is the trust boundary: no amount of LLM cleverness can override the physics.
"""

from __future__ import annotations

from typing import Any

from backend.gatekeeper import feasibility
from backend.gatekeeper.spec import get_model, get_tier
from backend.graph.state import GraphState


def gatekeeper_node(state: GraphState) -> dict[str, Any]:
    trace = list(state.get("trace", []))
    model_id = state.get("model_id")
    tier_id = state.get("tier_id")
    seq_len = int(state.get("seq_len", 4096))
    concurrency = int(state.get("concurrency", 1))
    dtype = state.get("dtype")
    prefix_ratio = float(state.get("prefix_ratio", 0.0))

    model = get_model(model_id) if model_id else None
    tier = get_tier(tier_id) if tier_id else None

    if model is None or tier is None:
        trace.append("gatekeeper: no model/tier constraints supplied; passing through")
        return {
            "feasible": True,
            "gate_report": {"note": "No hardware constraints supplied; feasibility not gated."},
            "gate_rejection": None,
            "trace": trace,
        }

    res = feasibility.evaluate_tier(
        tier, model, seq_len=seq_len, concurrency=concurrency, dtype=dtype, prefix_ratio=prefix_ratio
    )
    report = res.to_dict()
    report["tier_id"] = tier.id
    report["model_id"] = model.id

    if not res.feasible:
        rejection = (
            f"HARD REJECT (physics): {model.name} at seq_len={seq_len}, "
            f"concurrency={concurrency}, dtype={dtype or model.dtype_default} needs "
            f"{res.vram_total_gb:.1f} GB but {tier.name} has only {tier.memory_gb:.0f} GB "
            f"(short by {-res.headroom_gb:.1f} GB)."
        )
        trace.append("gatekeeper: INFEASIBLE -> hard reject")
        return {
            "feasible": False,
            "gate_report": report,
            "gate_rejection": rejection,
            "trace": trace,
        }

    trace.append(
        f"gatekeeper: feasible ({res.vram_total_gb:.1f} GB / {tier.memory_gb:.0f} GB, "
        f"{res.headroom_gb:.1f} GB headroom)"
    )
    return {
        "feasible": True,
        "gate_report": report,
        "gate_rejection": None,
        "trace": trace,
    }
