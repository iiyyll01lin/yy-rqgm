"""Held-out NEEDS loading + train/dev/val/test isolation (agent half).

A direct mirror of :mod:`backend.evaluator.anchors` for the agent's problem
statements. Data isolation (RQGM anti-reward-hacking), four disjoint splits:

* ``train`` — the GEPA program proposer's failure-trace / textual-gradient source
  (the frozen evaluator's red_flags on the champion program's architectures);
* ``dev``   — RANK/SELECT the agent Pareto frontier (``agent_evolve`` best);
* ``val``   — scored EXCLUSIVELY by the agent promotion gate
  (:func:`backend.agent.agent_evolve.evaluate_agent_promotion`);
* ``test``  — reserved purely for reporting (never selection, never gating).

Each need carries planted ``latent_flaws`` — the failure modes the generated
architecture will exhibit unless the champion program has a skill that covers
them (see :mod:`backend.inference.mock_scoring`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_NEEDS_PATH = _REPO_ROOT / "data" / "agent_tasks" / "needs.json"

TRAIN = "train"
DEV = "dev"
VAL = "val"
TEST = "test"
_VALID_SPLITS = {TRAIN, DEV, VAL, TEST}


def load_all_needs() -> list[dict[str, Any]]:
    if not _NEEDS_PATH.exists():
        return []
    with _NEEDS_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("needs", [])


def load_needs(split: str | None = None) -> list[dict[str, Any]]:
    """Return needs, optionally filtered to a single split.

    Needs with no ``split`` field default to ``train`` so legacy data feeds the
    proposer rather than silently leaking into the gate.
    """
    needs = load_all_needs()
    if split is None:
        return needs
    if split not in _VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_VALID_SPLITS}")
    return [n for n in needs if n.get("split", TRAIN) == split]


def split_counts() -> dict[str, int]:
    """``{split: n}`` for reporting."""
    out: dict[str, int] = {}
    for n in load_all_needs():
        sp = n.get("split", TRAIN)
        out[sp] = out.get(sp, 0) + 1
    return out
