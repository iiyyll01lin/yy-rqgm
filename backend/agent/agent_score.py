"""Agent utility scoring — the RQGM ASYMMETRIC signal.

A candidate agent PROGRAM is scored by running it on a set of needs to produce
concrete architectures, then scoring those architectures with the **epoch-FROZEN
champion EVALUATOR** (``versioning.get_champion_rubric_text`` →
``judge.score_candidate``). Utility aggregates ``mean(1 − deficit_loose)`` — the
champion's good-faith verdict on the agent's output.

THE INVARIANT (anti-reward-hacking asymmetry):

* the rubric text ALWAYS comes from the evaluator's champion versioning — the
  agent has no API to read or mutate it, and within an evaluator epoch it is
  frozen, so the agent cannot observe or game a moving judge;
* the agent is NEVER scored against the held-out needs' ground truth directly
  (that would let it fit the labels) — only the frozen evaluator's judgement
  counts. The evaluator, in turn, is gated by the held-out anchors (its own
  loop). Two different juries, by construction.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import mean
from typing import Any

from backend.agent.agent_generate import generate_architecture
from backend.agent.agent_program import AgentProgram
from backend.evaluator import versioning
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient, get_lemonade_client


@dataclass
class NeedScore:
    need_id: str
    domain: str | None
    architecture: str
    deficit_loose: float
    deficit_strict: float
    utility: float
    red_flags: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class ProgramScore:
    utility: float                                   # mean(1 - deficit_loose) over needs
    evaluator_epoch: int                             # the FROZEN champion epoch scored under
    champion_version: str                            # evaluator champion version (the jury)
    per_need_utility: dict[str, float] = field(default_factory=dict)
    per_domain_utility: dict[str, float] = field(default_factory=dict)
    need_scores: list[NeedScore] = field(default_factory=list)
    red_flag_counts: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "utility": round(self.utility, 6),
            "evaluator_epoch": self.evaluator_epoch,
            "champion_version": self.champion_version,
            "per_need_utility": {k: round(v, 6) for k, v in self.per_need_utility.items()},
            "per_domain_utility": {k: round(v, 6) for k, v in self.per_domain_utility.items()},
            "red_flag_counts": dict(self.red_flag_counts),
        }


def frozen_champion_rubric() -> str:
    """The evaluator's epoch-frozen champion rubric — the agent's ONLY jury.

    A named accessor so the asymmetry is explicit and grep-able: the agent half
    reads the evaluator champion here and NOWHERE mutates it.
    """
    return versioning.get_champion_rubric_text()


def score_program(
    program: AgentProgram,
    needs: list[dict[str, Any]],
    *,
    rubric_text: str | None = None,
    evaluator_epoch: int | None = None,
    client: LemonadeClient | None = None,
) -> ProgramScore:
    """Score ``program`` on ``needs`` under the FROZEN champion evaluator.

    ``rubric_text`` / ``evaluator_epoch`` default to the current evaluator champion
    (frozen within its epoch). Pass them explicitly only to RE-SCORE an archive
    under a *new* champion (selective erasure at an evaluator epoch boundary).
    """
    client = client or get_lemonade_client()
    if rubric_text is None:
        rubric_text = versioning.get_champion_rubric_text()
    if evaluator_epoch is None:
        evaluator_epoch = versioning.get_epoch()
    champion_version = versioning.get_champion_version()

    need_scores: list[NeedScore] = []
    per_domain: dict[str, list[float]] = {}
    red_flag_counts: dict[str, int] = {}

    for need in needs:
        domain_id = need.get("domain")
        latent = list(need.get("latent_flaws", []) or [])
        architecture, _proposal = generate_architecture(
            program, str(need.get("need", "")), domain_id=domain_id,
            need_flaws=latent, client=client,
        )
        scored = score_candidate(
            architecture, rubric_text, domain_id=domain_id, epoch=evaluator_epoch, client=client
        )
        util = 1.0 - float(scored["deficit_loose"])
        ns = NeedScore(
            need_id=str(need.get("id", "")),
            domain=domain_id,
            architecture=architecture,
            deficit_loose=float(scored["deficit_loose"]),
            deficit_strict=float(scored["deficit_strict"]),
            utility=util,
            red_flags=list(scored.get("red_flags", [])),
        )
        need_scores.append(ns)
        per_domain.setdefault(domain_id or "general", []).append(util)
        for rf in ns.red_flags:
            crit = str(rf.get("criterion", "unspecified"))
            red_flag_counts[crit] = red_flag_counts.get(crit, 0) + 1

    utility = mean([ns.utility for ns in need_scores]) if need_scores else 0.0
    return ProgramScore(
        utility=utility,
        evaluator_epoch=evaluator_epoch,
        champion_version=champion_version,
        per_need_utility={ns.need_id: ns.utility for ns in need_scores},
        per_domain_utility={d: mean(v) for d, v in per_domain.items()},
        need_scores=need_scores,
        red_flag_counts=red_flag_counts,
    )
