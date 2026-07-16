"""Held-out ground-truth anchor loading + train/dev/val/test isolation.

Data isolation (RQGM anti-reward-hacking), four disjoint splits:

* ``train`` — the GEPA proposer's failure-trace / textual-gradient source;
* ``dev``   — used to RANK/SELECT the Pareto frontier (``frontier.best`` by BBε).
  Selecting on a split the gate never sees removes the winner's curse: the gate
  is then a genuine held-out re-test rather than a re-test of the split model
  selection already peeked at;
* ``val``   — scored EXCLUSIVELY by the code gate (:func:`gate.evaluate_promotion`);
* ``test``  — reserved purely for reporting (never selection, never gating).

Each anchor carries planted ``flaws``/``strengths`` tags; :func:`anchor_candidate_text`
appends the machine-readable sentinel the offline mock reads (see
:mod:`backend.inference.mock_scoring`).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from backend.inference.mock_scoring import format_sentinel

_REPO_ROOT = Path(__file__).resolve().parents[2]
_ANCHOR_PATH = _REPO_ROOT / "data" / "anchor" / "anchor_architectures.json"

TRAIN = "train"
DEV = "dev"
VAL = "val"
TEST = "test"
_VALID_SPLITS = {TRAIN, DEV, VAL, TEST}


def load_all_anchors() -> list[dict[str, Any]]:
    if not _ANCHOR_PATH.exists():
        return []
    with _ANCHOR_PATH.open("r", encoding="utf-8") as fh:
        return json.load(fh).get("anchors", [])


def load_anchors(split: str | None = None) -> list[dict[str, Any]]:
    """Return anchors, optionally filtered to a single split.

    Anchors with no ``split`` field default to ``train`` so legacy data still
    feeds the proposer rather than silently leaking into the gate.
    """
    anchors = load_all_anchors()
    if split is None:
        return anchors
    if split not in _VALID_SPLITS:
        raise ValueError(f"unknown split {split!r}; expected one of {_VALID_SPLITS}")
    return [a for a in anchors if a.get("split", TRAIN) == split]


def split_counts() -> dict[str, dict[str, int]]:
    """``{split: {"weak": n, "strong": n, "total": n}}`` for reporting."""
    out: dict[str, dict[str, int]] = {}
    for a in load_all_anchors():
        sp = a.get("split", TRAIN)
        bucket = out.setdefault(sp, {"weak": 0, "strong": 0, "total": 0})
        bucket[a.get("label", "weak")] = bucket.get(a.get("label", "weak"), 0) + 1
        bucket["total"] += 1
    return out


def anchor_candidate_text(anchor: dict[str, Any]) -> str:
    """Architecture text + planted flaw/strength sentinel for offline scoring."""
    base = str(anchor.get("architecture", "")).strip()
    sentinel = format_sentinel(anchor.get("flaws"), anchor.get("strengths"))
    return f"{base}\n{sentinel}"


def weak(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in anchors if a.get("label") == "weak"]


def strong(anchors: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [a for a in anchors if a.get("label") == "strong"]
