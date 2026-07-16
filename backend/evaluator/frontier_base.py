"""Generic Pareto-frontier base shared by BOTH RQGM halves.

Factored out of ``backend/evaluator/frontier.py`` so the evaluator half (evolving
rubrics) and the agent half (evolving task-agent programs) share ONE correct
implementation of:

* multi-objective Pareto ``_dominates`` / ``_same_objectives``;
* a top-K non-dominated :class:`ParetoFrontier` container;
* Thompson-sampling (stochastic) and shallow-MCTS/UCT (deterministic) parent
  selection over the current non-dominated set.

The container is duck-typed over its members: it only needs ``.version``,
``.objectives``, ``.bbe``, ``.child_successes`` and ``.child_trials``. Each half
supplies its own member dataclass — the evaluator's carries ``rubric_text``, the
agent's carries a generic ``genome_text`` (the serialized program). The shared
:class:`FrontierMemberBase` gives that generic genome a home; the evaluator's
:class:`~backend.evaluator.frontier.FrontierMember` keeps its ``rubric_text``
field (unchanged behaviour) and exposes ``genome_text`` as an alias.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any


@dataclass
class FrontierMemberBase:
    """Generic frontier member: a genome + its multi-objective vector + bandit state."""

    version: str
    genome_text: str
    objectives: dict[str, float]
    bbe: float
    parent_version: str = ""
    # Thompson-sampling bandit state: how many children this member has parented
    # and how many were "successes" (survived Pareto + improved on the parent).
    child_successes: int = 0
    child_trials: int = 0

    def public_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "objectives": {k: round(v, 4) for k, v in self.objectives.items()},
            "bbe": round(self.bbe, 4),
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
    """Top-K non-dominated set of members (shared by evaluator + agent halves)."""

    def __init__(self, top_k: int = 8, *, prior_alpha: float = 1.0, prior_beta: float = 1.0):
        self.top_k = top_k
        self.members: list[Any] = []
        # Beta prior for the per-member Thompson-sampling parent selector.
        self._prior_alpha = prior_alpha
        self._prior_beta = prior_beta

    def add(self, member: Any) -> bool:
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

    def best(self) -> Any | None:
        """Best member by BBε-style separation lower bound (gate candidate)."""
        if not self.members:
            return None
        return max(self.members, key=lambda m: m.bbe)

    def sample_stochastic(self, rng: Any) -> Any:
        """Thompson-sample the next parent to mutate.

        Each member carries a Beta posterior on ``P(a child it parents improves)``
        — ``Beta(prior_alpha + successes, prior_beta + failures)``. We draw one
        sample per member and pick the arg-max (Thompson sampling), balancing
        exploiting productive parents against exploring under-tried ones. The
        Pareto filter is untouched: we only sample from the non-dominated set.
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

    def sample_uct(self, *, c: float = 1.4) -> Any:
        """Shallow-MCTS (UCT) parent selection — the deterministic alternative.

        Treats the non-dominated set as a shallow search tree: each member is a
        node whose reward is its improving-child rate. UCT balances exploiting the
        best success-rate parent against exploring under-tried ones
        (``exploit + c·sqrt(ln(N)/n)``), expanding any unvisited node first. Fully
        DETERMINISTIC (no RNG), so the frontier stays reproducible; like Thompson
        it only samples from the current Pareto set.
        """
        if not self.members:
            raise IndexError("empty frontier")
        if len(self.members) == 1:
            return self.members[0]
        total = sum(m.child_trials for m in self.members)
        for m in self.members:
            if m.child_trials == 0:
                return m  # expand an unvisited node first (standard UCT)
        best, best_score = self.members[0], float("-inf")
        for m in self.members:
            exploit = m.child_successes / m.child_trials
            explore = c * math.sqrt(math.log(max(total, 1)) / m.child_trials)
            score = exploit + explore
            if score > best_score:
                best_score, best = score, m
        return best

    def record_child_outcome(self, parent: Any, *, improved: bool) -> None:
        """Update ``parent``'s Beta posterior after evaluating one of its children."""
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


__all__ = ["FrontierMemberBase", "ParetoFrontier", "_dominates", "_same_objectives", "field"]
