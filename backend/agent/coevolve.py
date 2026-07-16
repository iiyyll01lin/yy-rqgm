"""Co-evolution scheduler: alternate the two RQGM halves under the asymmetry.

One round of full RQGM co-evolution (all offline/deterministic):

    1. AGENT step — the agent half evolves a program frontier against the
       epoch-FROZEN champion EVALUATOR (``agent_evolve.propose_agent_via_frontier``)
       and, if it clears the agent gate on held-out val needs, promotes it
       (agent epoch++). The evaluator is untouched here — within its epoch it is a
       fixed jury the agent cannot game.
    2. SELF-PLAY mining — the champion agent's concrete-architecture archive is
       mined for GAMED architectures (look strong under the champion's loose read
       but hide a poison pill it misses). These become adversarial samples.
    3. EVALUATOR step — the evaluator half evolves against the held-out ANCHORS
       PLUS the agent-derived gamed samples (its own held-out ground-truth jury),
       proposing a challenger rubric.
    4. BOUNDARY — optionally approve the evaluator challenger (code gate + HITL).
       On promotion the evaluator epoch advances, which TRIGGERS agent-utility
       selective erasure (evolve.approve_challenger re-scores the agent archive
       under the new champion). The asymmetry holds throughout: agent judged by
       the frozen evaluator, evaluator judged by held-out anchors.

Both manual endpoints (agent propose/promote; epoch propose/approve) still work
standalone; this just wires them into the alternating loop.
"""

from __future__ import annotations

from typing import Any

from backend.agent import agent_evolve, agent_versioning, concrete_archive
from backend.agent import needs as needs_ds
from backend.agent.agent_program import AgentProgram
from backend.evaluator import evolve as evaluator_evolve
from backend.evaluator import versioning
from backend.inference.lemonade_client import LemonadeClient, get_lemonade_client


def mine_agent_adversarial(
    program: AgentProgram | None = None,
    *,
    splits: tuple[str, ...] = (needs_ds.TRAIN, needs_ds.DEV),
    client: LemonadeClient | None = None,
) -> list[dict[str, Any]]:
    """Mine GAMED architectures from the champion agent (self-play adversarial pool).

    The agent, maximising utility under the frozen champion, produces architectures
    that game the champion's blind spots on kpi/drift needs — exactly the samples
    that should drive the evaluator to close those blind spots.
    """
    client = client or get_lemonade_client()
    program = program or agent_versioning.get_champion_program()
    pool_needs: list[dict[str, Any]] = []
    for sp in splits:
        pool_needs.extend(needs_ds.load_needs(sp))
    archives = concrete_archive.build_archive(pool_needs, program, client=client)
    return concrete_archive.adversarial_samples(archives)


def run_coevolution_round(
    *,
    agent_budget: int = 6,
    evaluator_budget: int = 6,
    seed: int = 1234,
    promote_agent: bool = True,
    approve_evaluator: bool = False,
    client: LemonadeClient | None = None,
) -> dict[str, Any]:
    """Run one alternating co-evolution round; return a structured summary.

    ``promote_agent`` gates the agent challenger through its own gate (frozen
    evaluator). ``approve_evaluator`` additionally approves the evaluator
    challenger (code gate + HITL) — which, on promotion, triggers agent-utility
    selective erasure. Defaults keep the evaluator promotion human-gated.
    """
    client = client or get_lemonade_client()

    # --- 1. AGENT step: evolve + gate against the FROZEN champion evaluator -----
    agent_proposal, agent_frontier = agent_evolve.propose_agent_via_frontier(
        budget=agent_budget, client=client, seed=seed
    )
    agent_promotion: dict[str, Any] = {}
    if agent_proposal and promote_agent:
        agent_promotion = agent_evolve.approve_agent_challenger(
            agent_proposal["challenger_id"], client=client
        )
    # Baseline the (possibly new) champion agent's val utility under the CURRENT
    # (still-frozen) evaluator, so the boundary erasure has a like-for-like before.
    champion_val_utility = agent_evolve.measure_champion_utility(client=client)

    # --- 2. SELF-PLAY mining: agent-derived gamed samples -----------------------
    agent_adversarial = mine_agent_adversarial(client=client)

    # --- 3. EVALUATOR step: evolve vs anchors + agent-derived gamed samples -----
    combined_adversarial = evaluator_evolve.generate_adversarial_pool() + agent_adversarial
    evaluator_proposal, evaluator_frontier = evaluator_evolve.propose_via_frontier(
        adversarial_samples=combined_adversarial, budget=evaluator_budget, client=client, seed=seed
    )

    # --- 4. BOUNDARY: optionally promote the evaluator -> triggers agent erasure -
    evaluator_promotion: dict[str, Any] = {}
    if approve_evaluator:
        evaluator_promotion = evaluator_evolve.approve_challenger(
            evaluator_proposal.version, approve=True, client=client
        )

    return {
        "agent_epoch": agent_versioning.get_agent_epoch(),
        "evaluator_epoch": versioning.get_epoch(),
        "agent": {
            "proposal": agent_proposal,
            "frontier": agent_frontier.to_dict(),
            "promotion": agent_promotion,
            "champion_val_utility": round(champion_val_utility, 6),
        },
        "self_play": {
            "agent_adversarial_samples": len(agent_adversarial),
            "targets": sorted({s["targets"] for s in agent_adversarial}),
        },
        "evaluator": {
            "challenger_id": evaluator_proposal.version,
            "frontier": evaluator_frontier.to_dict(),
            "adversarial_pool_size": len(combined_adversarial),
            "promotion": evaluator_promotion,
        },
        "asymmetry": (
            "agent scored by the epoch-frozen champion evaluator; evaluator gated by "
            "held-out anchors + agent-mined gamed samples; utilities erased at the boundary."
        ),
    }
