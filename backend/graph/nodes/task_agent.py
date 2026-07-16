"""Task Agent node: propose a workflow architecture for the domain need.

Uses the local model (Lemonade, mock fallback) and the routed template as a
scaffold. Emits a natural-language architecture + a structured proposal.

RQGM co-evolution: the agent persona/skills are no longer hard-coded here — the
hot path loads the epoch-CHAMPION agent PROGRAM from
:mod:`backend.agent.agent_versioning` (falling back to the immutable seed), so
improvements the agent half evolves are actually adopted in production.
"""

from __future__ import annotations

from typing import Any

from backend.agent import agent_versioning
from backend.agent.agent_generate import generate_architecture
from backend.agent.agent_program import load_seed_program
from backend.domains.registry import get_domain
from backend.graph.state import GraphState
from backend.inference.lemonade_client import get_lemonade_client


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
    # Hot path adopts the current CHAMPION agent program (fallback: immutable seed).
    try:
        program = agent_versioning.get_champion_program()
        program_version = agent_versioning.get_champion_version()
    except Exception:
        program, program_version = load_seed_program(), "agent-champion-0"

    architecture, proposal = generate_architecture(
        program,
        need,
        domain_id=domain_id,
        template_hint=_template_hint(domain_id, template_id),
        client=client,
    )

    trace.append(
        f"task_agent: proposed architecture ({len(architecture)} chars) "
        f"via champion program {program_version}"
    )
    return {
        "architecture": architecture,
        "proposal": proposal,
        "agent_program_version": program_version,
        "used_mock": client.using_mock,
        "trace": trace,
    }
