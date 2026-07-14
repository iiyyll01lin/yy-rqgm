"""Task Agent node: propose a workflow architecture for the domain need.

Uses the local model (Lemonade, mock fallback) and the routed template as a
scaffold. Emits a natural-language architecture + a structured proposal.
"""

from __future__ import annotations

from typing import Any

from backend.domains.registry import get_domain
from backend.graph.state import GraphState
from backend.inference.lemonade_client import MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json


def _template_hint(domain_id: str | None, template_id: str | None) -> str:
    if not (domain_id and template_id):
        return ""
    pack = get_domain(domain_id)
    if pack is None:
        return ""
    for tmpl in pack.workflow_templates():
        if tmpl.id == template_id:
            nodes = ", ".join(tmpl.nodes) if tmpl.nodes else "(propose nodes)"
            return (
                f"\nUse this template as a scaffold — '{tmpl.name}': {tmpl.description}\n"
                f"Suggested nodes: {nodes}\n"
            )
    return ""


def task_agent_node(state: GraphState) -> dict[str, Any]:
    need = state.get("need") or state.get("domain", "")
    domain_id = state.get("domain")
    template_id = state.get("matched_template_id")
    trace = list(state.get("trace", []))

    client = get_lemonade_client()
    system = (
        f"{MockMarker.TASK_AGENT.value}\n"
        "You are the Task Agent. Propose a concrete LangGraph agent architecture for the\n"
        "stated need. Favor clear State Schemas, separation of concerns, physical\n"
        "root-cause reasoning, and a deterministic safety gate for irreversible actions.\n"
        "Respond with STRICT JSON: {\"architecture\": str, \"nodes\": [str], "
        "\"state_schema\": {..}, \"tools\": [str], \"rationale\": str}."
    )
    user = f"Domain: {domain_id or 'general'}\nNeed: {need}{_template_hint(domain_id, template_id)}"

    raw = client.chat(
        [{"role": "system", "content": system}, {"role": "user", "content": user}],
        temperature=0.3,
        max_tokens=900,
    )
    proposal = extract_json(raw) or {}
    architecture = proposal.get("architecture") or raw.strip()

    trace.append(f"task_agent: proposed architecture ({len(architecture)} chars)")
    return {
        "architecture": architecture,
        "proposal": proposal,
        "used_mock": client.using_mock,
        "trace": trace,
    }
