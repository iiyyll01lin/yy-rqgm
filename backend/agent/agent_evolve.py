"""GEPA-style self-evolution for the agent PROGRAM (agent-half mirror of evolve.py).

Cold path (async, off the hot request path):

    1. PROPOSE  — maintain a Pareto FRONTIER of agent programs. Stochastically
       sample a parent, pick a ``train`` need it scores worst on, read the FROZEN
       champion evaluator's red_flags on that output as a *textual gradient*
       (GEPA), reflect→mutate the program (add the skill that closes the flagged
       gap), score the child on the ``dev`` selection needs under the frozen
       evaluator, and keep it only if non-dominated.
    2. GATE     — a challenger is promoted iff it beats the champion program on the
       held-out ``val`` needs, *as scored by the epoch-FROZEN champion EVALUATOR*
       (RQGM asymmetry: the agent's jury is the evaluator, never the needs' own
       ground truth). Promotion advances the AGENT epoch.
    3. ERASURE  — at an EVALUATOR epoch boundary the whole agent archive is
       re-scored under the NEW champion (utility selective erasure), mirroring the
       evaluator's memory selective erasure.

Everything runs deterministically offline via the inference mock.
"""

from __future__ import annotations

import hashlib
import random
from dataclasses import asdict, dataclass, field
from typing import Any

from backend.agent import agent_versioning
from backend.agent import needs as needs_ds
from backend.agent.agent_frontier import (
    AgentFrontierMember,
    compute_agent_objectives,
    persist_agent_frontier,
)
from backend.agent.agent_program import AgentProgram
from backend.agent.agent_score import score_program
from backend.evaluator import versioning
from backend.evaluator.frontier_base import ParetoFrontier
from backend.evaluator.gate import beta_binomial_superiority
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json


def _short_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:8]


# ---------------------------------------------------------------------------
# REFLECT + MUTATE (GEPA reflective step on the program)
# ---------------------------------------------------------------------------
def reflect_and_mutate_program(
    parent: AgentProgram,
    side_info: str,
    client: LemonadeClient,
) -> tuple[str, list[str], AgentProgram]:
    """Reflect on the frozen evaluator's red_flags; add the skill(s) that close them.

    Returns ``(reflection, proposed_changes, child_program)``. The child differs
    from the parent only by added skills — a mutation the champion evaluator can
    reward (it covers a flagged failure mode) WITHOUT the agent ever reading the
    rubric. If the flagged skill is already present the child equals the parent
    (a no-op the frontier's duplicate-objective filter rejects).
    """
    system = (
        f"{MockMarker.AGENT_MUTATE.value}\n"
        "You are a GEPA-style reflective optimizer improving an RQGM task-agent PROGRAM.\n"
        "Read the Actionable Side Information (the FROZEN champion evaluator's red_flags on\n"
        "this program's own architectures) as a textual gradient. Propose the minimal skill\n"
        "additions that would make the design survive those criticisms. Respond with STRICT\n"
        'JSON: {"reflection": str, "proposed_changes": [str], "add_skills": [str]}.'
    )
    user = (
        f"=== CURRENT AGENT PROGRAM ===\n{parent.genome_text()}\n\n"
        f"=== ACTIONABLE SIDE INFORMATION ===\n{side_info}\n"
    )
    parsed = extract_json(
        client.chat(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            temperature=0.4,
            max_tokens=600,
        )
    ) or {}
    reflection = str(parsed.get("reflection", "")) or "Reflective program mutation from frozen-evaluator feedback."
    proposed_changes = list(parsed.get("proposed_changes", []) or [])
    add_skills = [str(s).strip() for s in (parsed.get("add_skills", []) or []) if str(s).strip()]
    child = parent
    for skill in add_skills:
        child = child.with_skill(skill)
    return reflection, proposed_changes, child


def _select_failure_side_info(
    program: AgentProgram,
    train_needs: list[dict[str, Any]],
    *,
    rubric_text: str,
    evaluator_epoch: int,
    client: LemonadeClient,
) -> str:
    """Pick the ``train`` need the program scores WORST on; turn the frozen
    evaluator's red_flag criteria there into GEPA side-information."""
    ps = score_program(
        program, train_needs, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client
    )
    if not ps.need_scores:
        return "- (no train needs available; sharpen an existing skill)"
    worst = min(ps.need_scores, key=lambda n: n.utility)
    crits = [str(rf.get("criterion", "")) for rf in worst.red_flags if rf.get("criterion")]
    if not crits:
        return (
            f"- Evaluator trace: program output for need '{worst.need_id}' already clears the "
            "champion evaluator's criteria; no new gap to close."
        )
    # Include the red-flag DETAILS (which name the specific planted flaw), because a
    # single criterion can catch several flaws — the flaw name lets the reflective
    # mutator add the RIGHT skill (mirrors GEPA reading the failure trace, not just
    # the criterion label).
    details = "; ".join(str(rf.get("detail", "")) for rf in worst.red_flags if rf.get("detail"))
    return (
        f"- Evaluator trace: the FROZEN champion evaluator flagged the program's architecture for "
        f"need '{worst.need_id}' (utility {worst.utility:.2f}). Red-flag criteria: {', '.join(crits)}. "
        f"Findings: {details}"
    )


# ---------------------------------------------------------------------------
# PROPOSE via Pareto frontier (GEPA population search over programs)
# ---------------------------------------------------------------------------
def agent_gepa_evolve(
    incumbent: AgentProgram,
    *,
    agent_epoch: int,
    rubric_text: str,
    evaluator_epoch: int,
    budget: int = 6,
    top_k: int = 8,
    client: LemonadeClient | None = None,
    seed: int = 1234,
    strategy: str = "thompson",
) -> ParetoFrontier:
    """GEPA budget loop over a Pareto frontier of agent programs.

    Selection is on the ``dev`` needs (``val`` reserved for the gate). The frozen
    champion evaluator scores every candidate — the agent never reads the rubric.
    """
    client = client or get_lemonade_client()
    dev_needs = needs_ds.load_needs(needs_ds.DEV)
    train_needs = needs_ds.load_needs(needs_ds.TRAIN)
    rng = random.Random(seed)

    frontier = ParetoFrontier(top_k=top_k)
    inc_obj, inc_bbe, inc_pn = compute_agent_objectives(
        incumbent, dev_needs, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client
    )
    frontier.add(
        AgentFrontierMember(
            version=agent_versioning.get_champion_version(),
            genome_text=incumbent.genome_text(),
            objectives=inc_obj, bbe=inc_bbe, parent_version="",
            program=incumbent.to_dict(), per_need_utility=inc_pn, utility=inc_bbe,
        )
    )

    for i in range(budget):
        parent = frontier.sample_uct() if strategy == "mcts" else frontier.sample_stochastic(rng)
        parent_program = AgentProgram.from_dict(parent.program)
        side_info = _select_failure_side_info(
            parent_program, train_needs, rubric_text=rubric_text,
            evaluator_epoch=evaluator_epoch, client=client,
        )
        _refl, _changes, child_program = reflect_and_mutate_program(parent_program, side_info, client)
        child_version = f"agent-challenger-e{agent_epoch}-{_short_hash(parent.version + side_info + str(i))}"
        obj, bbe, per_need = compute_agent_objectives(
            child_program, dev_needs, rubric_text=rubric_text,
            evaluator_epoch=evaluator_epoch, client=client,
        )
        kept = frontier.add(
            AgentFrontierMember(
                version=child_version, genome_text=child_program.genome_text(),
                objectives=obj, bbe=bbe, parent_version=parent.version,
                program=child_program.to_dict(), per_need_utility=per_need, utility=bbe,
            )
        )
        # "Success" for the parent: a child that survived the Pareto filter AND
        # improved its mean dev utility (the same signal the gate scores).
        frontier.record_child_outcome(parent, improved=bool(kept and bbe > parent.bbe))

    return frontier


def propose_agent_via_frontier(
    *,
    budget: int = 6,
    client: LemonadeClient | None = None,
    seed: int = 1234,
    strategy: str = "thompson",
) -> tuple[dict[str, Any], ParetoFrontier]:
    """Run ``agent_gepa_evolve`` and register the frontier's best member as the
    challenger handed to the agent gate. Persists the whole frontier for repro.

    Returns ``(proposal_dict, frontier)``. ``proposal_dict`` is empty (no
    challenger) when the frontier never beat the incumbent champion program.
    """
    client = client or get_lemonade_client()
    agent_epoch = agent_versioning.get_agent_epoch()
    evaluator_epoch = versioning.get_epoch()
    rubric_text = versioning.get_champion_rubric_text()  # FROZEN champion evaluator
    champion_program = agent_versioning.get_champion_program()
    champion_version = agent_versioning.get_champion_version()

    frontier = agent_gepa_evolve(
        champion_program, agent_epoch=agent_epoch, rubric_text=rubric_text,
        evaluator_epoch=evaluator_epoch, budget=budget, client=client,
        seed=seed, strategy=strategy,
    )
    try:
        persist_agent_frontier(frontier, agent_epoch)
    except Exception:
        pass

    best = frontier.best()
    if best is None or best.version == champion_version:
        return {}, frontier

    best_program = best.to_program()
    metrics = {
        "split": "dev",
        "frontier_size": len(frontier.members),
        "skills": list(best_program.skills),
        "added_skills": sorted(set(best_program.skills) - set(champion_program.skills)),
        "champion_dev_utility": round(
            next((m.bbe for m in frontier.members if m.version == champion_version), 0.0), 6
        ),
        "challenger_dev_utility": round(best.bbe, 6),
        "utility": round(best.bbe, 6),
        "objectives": {k: round(v, 4) for k, v in best.objectives.items()},
        "note": (
            "best of the agent Pareto frontier on the dev selection needs (mean utility under "
            "the FROZEN champion evaluator); the agent gate re-checks on the held-out val needs."
        ),
    }
    agent_versioning.register_challenger(
        version=best.version, program=best_program, metrics=metrics,
        parent_version=champion_version, evaluator_epoch=evaluator_epoch,
    )
    return {"challenger_id": best.version, "metrics": metrics}, frontier


# ---------------------------------------------------------------------------
# AGENT GATE (challenger vs champion on held-out val, scored by frozen evaluator)
# ---------------------------------------------------------------------------
@dataclass
class AgentGateConfig:
    """Tunable strictness for the agent promotion gate (mirrors evaluator GateConfig)."""

    min_utility_margin: float = 0.0          # P1: tie favours the incumbent program
    posterior_threshold: float = 0.9         # P2: promote iff P(Δutil>0) >= this
    min_detectable_effect: float = 0.02      # P2: require mean Δutility >= this (MDE)
    prior_alpha: float = 1.0
    prior_beta: float = 1.0
    win_epsilon: float = 1e-6


@dataclass
class AgentGateResult:
    passed: bool
    reason: str
    champion_utility: float
    challenger_utility: float
    utility_delta: float
    p1_non_inferior: bool
    p2_passed: bool
    posterior_prob_improvement: float
    posterior_threshold: float
    effect_size: float
    min_detectable_effect: float
    n_wins: int
    n_losses: int
    n_ties: int
    n_val: int
    evaluator_epoch: int
    champion_version: str
    per_need_delta: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def evaluate_agent_promotion(
    version: str,
    *,
    rubric_text: str | None = None,
    client: LemonadeClient | None = None,
    config: AgentGateConfig | None = None,
) -> AgentGateResult:
    """Agent gate: challenger beats champion program on held-out ``val`` needs, as
    scored by the epoch-FROZEN champion EVALUATOR (never the needs' ground truth)."""
    config = config or AgentGateConfig()
    client = client or get_lemonade_client()
    if rubric_text is None:
        rubric_text = versioning.get_champion_rubric_text()  # FROZEN champion evaluator
    evaluator_epoch = versioning.get_epoch()

    challenger_program = agent_versioning.get_challenger_program(version)
    if challenger_program is None:
        raise KeyError(f"unknown agent challenger: {version}")
    champion_program = agent_versioning.get_champion_program()
    val_needs = needs_ds.load_needs(needs_ds.VAL)

    champ = score_program(champion_program, val_needs, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client)
    chal = score_program(challenger_program, val_needs, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client)
    delta = chal.utility - champ.utility

    wins = losses = ties = 0
    per_need_delta: dict[str, float] = {}
    for nid, cu in champ.per_need_utility.items():
        d = chal.per_need_utility.get(nid, 0.0) - cu
        per_need_delta[nid] = round(d, 6)
        if d > config.win_epsilon:
            wins += 1
        elif d < -config.win_epsilon:
            losses += 1
        else:
            ties += 1

    posterior = beta_binomial_superiority(
        wins, losses, prior_alpha=config.prior_alpha, prior_beta=config.prior_beta
    )
    p1 = delta > config.min_utility_margin
    p2 = posterior >= config.posterior_threshold and delta >= config.min_detectable_effect
    passed = p1 and p2

    if passed:
        reason = (
            f"agent gate PASSED: challenger utility {chal.utility:.4f} > champion {champ.utility:.4f} "
            f"(delta {delta:+.4f}) on {wins}W/{losses}L/{ties}T val needs; posterior "
            f"P(Δutil>0)={posterior:.4f} >= {config.posterior_threshold}, effect >= MDE "
            f"{config.min_detectable_effect}. Jury = FROZEN champion evaluator {champ.champion_version}."
        )
    elif not p1:
        reason = (
            f"agent gate FAILED (P1 non-inferiority): challenger utility {chal.utility:.4f} does not "
            f"exceed champion {champ.utility:.4f} (delta {delta:+.4f}); tie favours the incumbent program."
        )
    else:
        reason = (
            f"agent gate FAILED (P2): posterior P(Δutil>0)={posterior:.4f} (need "
            f">= {config.posterior_threshold}) / effect {delta:.4f} (need >= "
            f"{config.min_detectable_effect}) on {wins}W/{losses}L/{ties}T val needs."
        )

    return AgentGateResult(
        passed=passed, reason=reason,
        champion_utility=round(champ.utility, 4),
        challenger_utility=round(chal.utility, 4),
        utility_delta=round(delta, 4),
        p1_non_inferior=p1, p2_passed=p2,
        posterior_prob_improvement=round(posterior, 4),
        posterior_threshold=config.posterior_threshold,
        effect_size=round(delta, 4),
        min_detectable_effect=config.min_detectable_effect,
        n_wins=wins, n_losses=losses, n_ties=ties, n_val=len(val_needs),
        evaluator_epoch=evaluator_epoch,
        champion_version=champ.champion_version,
        per_need_delta=per_need_delta,
    )


def approve_agent_challenger(
    version: str,
    *,
    client: LemonadeClient | None = None,
    config: AgentGateConfig | None = None,
) -> dict[str, Any]:
    """Promote ``version`` to champion PROGRAM iff the agent gate passes (agent epoch++).

    There is no HITL veto on the agent half: the gate IS the frozen champion
    evaluator's held-out verdict (the RQGM ground-truth jury for the agent).
    """
    gate = evaluate_agent_promotion(version, client=client, config=config)
    if not gate.passed:
        return {
            "agent_epoch_id": agent_versioning.get_agent_epoch(),
            "applied": False,
            "champion_version": agent_versioning.get_champion_version(),
            "gate": gate.to_dict(),
            "reason": gate.reason,
        }
    result = agent_versioning.promote_champion_program(version)
    return {
        "agent_epoch_id": result["agent_epoch_id"],
        "applied": True,
        "champion_version": result["champion_version"],
        "prior_epoch": result["prior_epoch"],
        "gate": gate.to_dict(),
        "reason": gate.reason,
    }


def measure_champion_utility(
    *,
    rubric_text: str | None = None,
    client: LemonadeClient | None = None,
    record: bool = True,
) -> float:
    """Score the champion PROGRAM on the ``val`` needs under the current (frozen)
    champion evaluator and (by default) persist it as the utility baseline.

    Called before an evaluator epoch boundary so :func:`selective_erasure_agent`
    has a like-for-like (val-measured) ``before`` to compare the re-score against.
    """
    client = client or get_lemonade_client()
    if rubric_text is None:
        rubric_text = versioning.get_champion_rubric_text()
    evaluator_epoch = versioning.get_epoch()
    val_needs = needs_ds.load_needs(needs_ds.VAL)
    champ = score_program(
        agent_versioning.get_champion_program(), val_needs,
        rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client,
    )
    if record:
        agent_versioning.record_champion_utility(champ.utility, evaluator_epoch)
    return champ.utility


# ---------------------------------------------------------------------------
# SELECTIVE ERASURE of agent utility at an EVALUATOR epoch boundary
# ---------------------------------------------------------------------------
def selective_erasure_agent(
    new_champion_text: str,
    new_evaluator_epoch: int,
    *,
    client: LemonadeClient | None = None,
) -> dict[str, Any]:
    """Re-score the whole agent archive under the NEW champion evaluator.

    The evaluator half just promoted, so every agent utility measured under the
    PRIOR evaluator epoch is now stale (it was the old jury's verdict). We
    re-score the champion program + all challengers on the ``val`` needs under the
    new champion and re-stamp their utilities + evaluator epoch — the agent-half
    mirror of the evaluator's memory selective erasure. A program that only looked
    good because it gamed a prior champion blind spot loses utility here.
    """
    client = client or get_lemonade_client()
    val_needs = needs_ds.load_needs(needs_ds.VAL)

    prior_champ_util, _prior_ev = agent_versioning.get_champion_utility()
    champion_program = agent_versioning.get_champion_program()
    champ = score_program(
        champion_program, val_needs, rubric_text=new_champion_text,
        evaluator_epoch=new_evaluator_epoch, client=client,
    )
    agent_versioning.record_champion_utility(champ.utility, new_evaluator_epoch)

    rescored = 1
    dropped = 0
    if prior_champ_util is not None and champ.utility < prior_champ_util - 1e-9:
        dropped += 1
    for entry in agent_versioning.list_challengers():
        prog = agent_versioning.get_challenger_program(entry["version"])
        if prog is None:
            continue
        prior_util = (entry.get("metrics", {}) or {}).get("utility")
        sc = score_program(
            prog, val_needs, rubric_text=new_champion_text,
            evaluator_epoch=new_evaluator_epoch, client=client,
        )
        agent_versioning.update_challenger_utility(entry["version"], sc.utility, new_evaluator_epoch)
        rescored += 1
        if prior_util is not None and sc.utility < float(prior_util) - 1e-9:
            dropped += 1

    return {
        "rescored": rescored,
        "utilities_dropped": dropped,
        "new_evaluator_epoch": new_evaluator_epoch,
        "champion_utility_before": round(prior_champ_util, 6) if prior_champ_util is not None else None,
        "champion_utility_after": round(champ.utility, 6),
        "note": (
            "agent utilities re-scored under the new champion evaluator (selective erasure); "
            "programs that gamed a prior champion blind spot lose utility."
        ),
    }
