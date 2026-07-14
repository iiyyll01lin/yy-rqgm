"""Router node: map the need to candidate workflow templates."""

from __future__ import annotations

from typing import Any

from backend.graph.router import route_need
from backend.graph.state import GraphState


def router_node(state: GraphState) -> dict[str, Any]:
    need = state.get("need") or state.get("domain", "")
    domain_id = state.get("domain")
    trace = list(state.get("trace", []))

    if not need:
        trace.append("router: no need provided; skipped")
        return {"trace": trace}

    matches = route_need(need, domain_id=domain_id, top_k=5)
    matched_templates = [m.to_public_dict() for m in matches]
    best = matches[0].template.id if matches else None
    trace.append(f"router: matched {len(matches)} template(s); best={best}")

    return {
        "matched_templates": matched_templates,
        "matched_template_id": best,
        "trace": trace,
    }
