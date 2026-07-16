"""Program-driven architecture generation (the agent's hot path + scoring path).

Both the hot task-agent node and the offline agent-evolution loop generate a
concrete architecture from an :class:`~backend.agent.agent_program.AgentProgram`
here, so the two paths never drift. Offline (mock) the generation is
deterministic and *program-aware* (a better program yields a better
architecture); live it is a normal model call whose ``response_format`` the
server may honour. The frozen champion evaluator scores the result — never these
generation internals.
"""

from __future__ import annotations

from typing import Any

from backend.agent.agent_program import AgentProgram
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json

# The task-agent's structured-output contract (moved out of the graph node so the
# evolving persona lives in the PROGRAM while the output shape stays invariant).
_JSON_CONTRACT = (
    'Respond with STRICT JSON: {"architecture": str, "nodes": [str], '
    '"state_schema": {..}, "tools": [str], "rationale": str}.'
)


def build_generation_messages(
    program: AgentProgram,
    need: str,
    domain_id: str | None = None,
    *,
    need_flaws: list[str] | None = None,
    template_hint: str = "",
) -> list[dict[str, str]]:
    """Compose (system, user) messages for a program-driven generation.

    The ``[[agent_skills:...]]`` sentinel (and, on the offline scoring path, the
    ``[[need_flaws:...]]`` sentinel) let the deterministic mock resolve the
    architecture's planted flaws/strengths from the program's skills. A LIVE model
    ignores the sentinels and designs from the natural-language persona.
    """
    system = (
        f"{MockMarker.TASK_AGENT.value}\n"
        f"{program.render_system_prompt(domain_id)}\n"
        f"{_JSON_CONTRACT}\n"
        f"{program.skill_sentinel()}"
    )
    if need_flaws is not None:
        system += f"\n[[need_flaws:{','.join(need_flaws)}]]"
    user = f"Domain: {domain_id or 'general'}\nNeed: {need}{template_hint}"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def generate_architecture(
    program: AgentProgram,
    need: str,
    *,
    domain_id: str | None = None,
    need_flaws: list[str] | None = None,
    template_hint: str = "",
    client: LemonadeClient | None = None,
) -> tuple[str, dict[str, Any]]:
    """Run ``program`` on ``need``; return ``(architecture_text, proposal_dict)``.

    ``need_flaws`` (offline scoring path) plants the need's latent failure modes so
    the mock can resolve which ones the program's skills cover.
    """
    client = client or get_lemonade_client()
    messages = build_generation_messages(
        program, need, domain_id, need_flaws=need_flaws, template_hint=template_hint
    )
    raw = client.chat(messages, temperature=0.3, max_tokens=900)
    proposal = extract_json(raw) or {}
    architecture = proposal.get("architecture") or raw.strip()
    return architecture, proposal
