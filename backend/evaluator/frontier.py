"""Pareto frontier for GEPA population search over evaluator rubrics.

Instead of a single mutation lineage (the shipped ``propose_challenger``), the
proposer maintains a **population** of challenger rubrics and keeps the top-K
*non-dominated* members across multiple objectives (EvoSkill-style Pareto
frontier). Objectives:

* ``sep::<criterion>`` — per-criterion weak-vs-strong separation on the SELECTION
  (``dev``) split, derived from the judge's per-criterion penalties (rewards a
  rubric that adds coverage for a specific failure mode / poison pill);
* ``adversarial`` — separation on self-play red-team samples (Phase 3): rubrics
  that stay stringent on gamed architectures score higher;
* ``parsimony`` — ``-(# GEPA-added criteria)``: keeps a real trade-off on the
  frontier (broad coverage vs. minimal rubric) so evolution does not collapse to
  a single lineage.

The frontier's best member (by BBε-style separation lower bound on the ``dev``
selection split) is the one handed to the code gate. Selection is deliberately
done on ``dev`` — a split the gate never sees — so ``gate.evaluate_promotion``
(on ``val``) is a genuine held-out re-test, not a re-test of the split the
selector already peeked at (no winner's curse). The whole frontier is persisted
to ``data/frontier`` for reproducibility.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator.frontier_base import (
    ParetoFrontier,
    _dominates,
    _same_objectives,
)
from backend.evaluator.gate import separation_lower_bound
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient

# Re-exported for backward compatibility: the generic Pareto container + dominance
# helpers now live in :mod:`backend.evaluator.frontier_base` (shared with the agent
# half). The evaluator's behaviour is unchanged — same classes, same semantics.
__all__ = [
    "FrontierMember",
    "ParetoFrontier",
    "_dominates",
    "_same_objectives",
    "compute_objectives",
    "persist_frontier",
    "PARSIMONY",
    "ADVERSARIAL",
]

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTIER_DIR = _REPO_ROOT / "data" / "frontier"

PARSIMONY = "parsimony"
ADVERSARIAL = "adversarial"


@dataclass
class FrontierMember:
    version: str
    rubric_text: str
    objectives: dict[str, float]
    bbe: float
    added_criteria: list[str] = field(default_factory=list)
    parent_version: str = ""
    sel_deficits: dict[str, float] = field(default_factory=dict)
    # Thompson-sampling bandit state: how many children this member has parented
    # and how many of them were gate-improving (survived Pareto + raised BBε).
    child_successes: int = 0
    child_trials: int = 0

    @property
    def genome_text(self) -> str:
        """Generic-genome alias (the evaluator's genome IS its rubric text)."""
        return self.rubric_text

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "objectives": {k: round(v, 4) for k, v in self.objectives.items()},
            "bbe": round(self.bbe, 4),
            "added_criteria": self.added_criteria,
            "parent_version": self.parent_version,
            "child_successes": self.child_successes,
            "child_trials": self.child_trials,
        }


# The generic Pareto container + dominance helpers now live in ``frontier_base``
# (shared with the agent half) and are imported/re-exported above; the evaluator's
# ``FrontierMember`` (rubric_text) is unchanged, so behaviour is byte-identical.


# ---------------------------------------------------------------------------
# Objective computation
# ---------------------------------------------------------------------------
def compute_objectives(
    rubric_text: str,
    sel_anchors: list[dict[str, Any]],
    *,
    domain_id: str | None = "smart_manufacturing",
    epoch: int = 0,
    client: LemonadeClient | None = None,
    added_criteria: list[str] | None = None,
    adversarial_samples: list[dict[str, Any]] | None = None,
) -> tuple[dict[str, float], float, dict[str, float]]:
    """Return ``(objectives, bbe, sel_loose_deficits)`` for one rubric.

    ``sel_anchors`` is the SELECTION split (``dev``) — never ``val`` (reserved for
    the gate) or ``test`` (reporting). ``objectives`` = per-criterion separation
    (``sep::<id>``) + ``parsimony`` + (if ``adversarial_samples`` given) an
    ``adversarial`` separation objective; ``bbe`` is the BBε lower bound of the
    overall separation on ``sel_anchors``.
    """
    weak = anchor_ds.weak(sel_anchors)
    strong = anchor_ds.strong(sel_anchors)
    per_anchor: dict[str, dict[str, Any]] = {}
    loose: dict[str, float] = {}
    for a in sel_anchors:
        res = score_candidate(
            anchor_ds.anchor_candidate_text(a), rubric_text,
            domain_id=a.get("domain"), epoch=epoch, client=client,
        )
        per_anchor[a["id"]] = res
        loose[a["id"]] = res["deficit_loose"]

    criteria_ids = {c for r in per_anchor.values() for c in r["criterion_penalties"]}
    objectives: dict[str, float] = {}
    for cid in criteria_ids:
        w = mean([per_anchor[a["id"]]["criterion_penalties"].get(cid, 0.0) for a in weak]) if weak else 0.0
        s = mean([per_anchor[a["id"]]["criterion_penalties"].get(cid, 0.0) for a in strong]) if strong else 0.0
        objectives[f"sep::{cid}"] = w - s

    objectives[PARSIMONY] = -float(len(added_criteria or []))

    if adversarial_samples:
        adv_weak = [
            score_candidate(anchor_ds.anchor_candidate_text(a), rubric_text,
                            domain_id=a.get("domain"), epoch=epoch, client=client)["deficit_loose"]
            for a in adversarial_samples
        ]
        # Reward staying STRINGENT (high deficit) on gamed samples.
        objectives[ADVERSARIAL] = mean(adv_weak) if adv_weak else 0.0

    bbe = separation_lower_bound(loose, sel_anchors)
    return objectives, bbe, loose


# ---------------------------------------------------------------------------
# Persistence (reproducibility)
# ---------------------------------------------------------------------------
def persist_frontier(frontier: ParetoFrontier, epoch: int, *, directory: Path | None = None) -> str:
    """Write the frontier (rubric text of every member) to ``data/frontier``."""
    directory = directory or _FRONTIER_DIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"epoch-{epoch}.json"
    payload = {
        "epoch": epoch,
        "created_at": int(time.time()),
        "summary": frontier.to_dict(),
        "members": [
            {**m.public_dict(), "rubric_text": m.rubric_text, "sel_deficits": m.sel_deficits}
            for m in frontier.members
        ],
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return str(path)
