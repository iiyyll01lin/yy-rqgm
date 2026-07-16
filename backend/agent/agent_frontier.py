"""Pareto frontier for GEPA population search over agent PROGRAMS.

The agent half's mirror of ``backend/evaluator/frontier.py``. It reuses the
generic Pareto container + Thompson/UCT selection from
:mod:`backend.evaluator.frontier_base`; only the member payload and the
objectives differ (a program genome scored by the FROZEN champion evaluator, not
a rubric scored by held-out anchors).

Objectives (all maximised):

* ``util::<domain>`` — mean agent utility on the ``dev`` selection needs of that
  domain (``mean(1 − deficit_loose)`` under the frozen champion evaluator);
* ``parsimony``     — ``-(# skills)``: keeps a real trade-off on the frontier
  (broad capability vs. a minimal program) so evolution does not just pile on
  skills. An irrelevant skill is a no-op on utility but costs parsimony, so it is
  dominated — the offline honesty control against no-op program bloat.

Selection is on ``dev`` — a split the agent gate (on ``val``) never sees — so the
gate is a genuine held-out re-test (no winner's curse), exactly as on the
evaluator half.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from backend.agent.agent_program import AgentProgram
from backend.agent.agent_score import score_program
from backend.evaluator.frontier_base import FrontierMemberBase, ParetoFrontier
from backend.inference.lemonade_client import LemonadeClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_AGENT_FRONTIER_DIR = _REPO_ROOT / "data" / "agent_frontier"

PARSIMONY = "parsimony"


@dataclass
class AgentFrontierMember(FrontierMemberBase):
    """A frontier member wrapping an agent program (``genome_text`` = serialized)."""

    program: dict[str, Any] = field(default_factory=dict)
    per_need_utility: dict[str, float] = field(default_factory=dict)
    utility: float = 0.0

    def public_dict(self) -> dict[str, Any]:
        d = super().public_dict()
        d["skills"] = list(self.program.get("skills", []))
        d["utility"] = round(self.utility, 4)
        return d

    def to_program(self) -> AgentProgram:
        return AgentProgram.from_dict(self.program)


def compute_agent_objectives(
    program: AgentProgram,
    dev_needs: list[dict[str, Any]],
    *,
    rubric_text: str,
    evaluator_epoch: int,
    client: LemonadeClient | None = None,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Return ``(objectives, bbe, per_need_utility)`` for one program on ``dev``.

    ``bbe`` is the mean dev utility (the scalar the agent gate's ``best`` uses);
    ``objectives`` are per-domain utility + parsimony (the Pareto trade-off).
    """
    score = score_program(
        program, dev_needs, rubric_text=rubric_text, evaluator_epoch=evaluator_epoch, client=client
    )
    objectives: dict[str, float] = {
        f"util::{domain}": u for domain, u in score.per_domain_utility.items()
    }
    objectives[PARSIMONY] = -float(len(set(program.skills)))
    return objectives, score.utility, score.per_need_utility


def persist_agent_frontier(
    frontier: ParetoFrontier, agent_epoch: int, *, directory: Path | None = None
) -> str:
    """Write the agent frontier (program of every member) to ``data/agent_frontier``."""
    directory = directory or _AGENT_FRONTIER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch-{agent_epoch}.json"
    payload = {
        "agent_epoch": agent_epoch,
        "created_at": int(time.time()),
        "summary": frontier.to_dict(),
        "members": [
            {**m.public_dict(), "program": m.program, "per_need_utility": m.per_need_utility}
            for m in frontier.members
        ],
    }
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(path)
