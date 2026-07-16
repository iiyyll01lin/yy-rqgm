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
from backend.evaluator.gate import separation_lower_bound
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient

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


def _dominates(a: dict[str, float], b: dict[str, float]) -> bool:
    """True iff objective vector ``a`` Pareto-dominates ``b`` (maximisation)."""
    keys = set(a) | set(b)
    at_least_one_better = False
    for k in keys:
        av, bv = a.get(k, 0.0), b.get(k, 0.0)
        if av < bv - 1e-9:
            return False
        if av > bv + 1e-9:
            at_least_one_better = True
    return at_least_one_better


def _same_objectives(a: dict[str, float], b: dict[str, float]) -> bool:
    keys = set(a) | set(b)
    return all(abs(a.get(k, 0.0) - b.get(k, 0.0)) <= 1e-9 for k in keys)


class ParetoFrontier:
    """Top-K non-dominated set of challenger rubrics."""

    def __init__(self, top_k: int = 8, *, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.top_k = top_k
        self.members: list[FrontierMember] = []
        # Beta prior for the per-member Thompson-sampling parent selector.
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    def add(self, member: FrontierMember) -> bool:
        """Add ``member``; drop anything it dominates. Returns True if kept."""
        for existing in self.members:
            if existing.version == member.version:
                continue
            if _dominates(existing.objectives, member.objectives):
                return False  # dominated by an incumbent frontier member
            if _same_objectives(existing.objectives, member.objectives):
                return False  # exact duplicate objective vector; keep the frontier lean
        # Remove members dominated by the newcomer.
        self.members = [
            m for m in self.members if not _dominates(member.objectives, m.objectives)
        ]
        self.members.append(member)
        if len(self.members) > self.top_k:
            self.members.sort(key=lambda m: m.bbe, reverse=True)
            self.members = self.members[: self.top_k]
        return True

    def best(self) -> FrontierMember | None:
        """Best member by BBε-style ``dev`` separation lower bound (gate candidate)."""
        if not self.members:
            return None
        return max(self.members, key=lambda m: m.bbe)

    def sample_stochastic(self, rng: Any) -> FrontierMember:
        """Thompson-sample the next parent to mutate.

        Each frontier member carries a Beta posterior on ``P(a child it parents is
        gate-improving)`` — ``Beta(prior_alpha + successes, prior_beta + failures)``.
        We draw one sample per member and pick the arg-max (Thompson sampling),
        which balances exploiting parents that have produced improving children
        against exploring under-tried ones. This replaces the old near-uniform
        ``bbe + 1.0`` weighting (``BBε ∈ ~[0, 0.5]`` made every weight ≈ 1, i.e.
        effectively uniform). The Pareto filter is untouched: we only ever sample
        from the current non-dominated set (``self.members``).
        """
        if len(self.members) == 1:
            return self.members[0]
        best_member = self.members[0]
        best_draw = -1.0
        for m in self.members:
            a = self._prior_alpha + m.child_successes
            b = self._prior_beta + (m.child_trials - m.child_successes)
            draw = rng.betavariate(a, b)
            if draw > best_draw:
                best_draw = draw
                best_member = m
        return best_member

    def record_child_outcome(self, parent: FrontierMember, *, improved: bool) -> None:
        """Update ``parent``'s Beta posterior after evaluating one of its children.

        ``improved`` is the gate-aligned success signal used by the caller: a
        child that survived the Pareto filter AND raised the BBε separation lower
        bound over its parent. Ties/duplicates/malformed mutations are failures.
        """
        parent.child_trials += 1
        if improved:
            parent.child_successes += 1

    def to_dict(self) -> dict[str, Any]:
        best = self.best()
        return {
            "size": len(self.members),
            "top_k": self.top_k,
            "objectives": sorted({k for m in self.members for k in m.objectives}),
            "best_version": best.version if best else None,
            "members": [m.public_dict() for m in self.members],
        }


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
