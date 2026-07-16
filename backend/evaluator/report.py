"""RQGM transparency report: val/test separation, hack ratio, judge agreement.

Backing data for the ``GET /api/admin/report`` endpoint and the compact summary
added to ``/health``. Everything is computed on held-out anchors under the
current champion so the frontend / docs can show honest, reproducible numbers.
"""

from __future__ import annotations

import json
from pathlib import Path
from statistics import mean
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import panel, rqgm_adapter, versioning
from backend.evaluator.gate import _separation
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTIER_DIR = _REPO_ROOT / "data" / "frontier"


def _split_separation(
    rubric_text: str, split: str, epoch: int, client: LemonadeClient | None
) -> dict[str, Any]:
    anchors = anchor_ds.load_anchors(split)
    deficits = {
        a["id"]: score_candidate(
            anchor_ds.anchor_candidate_text(a), rubric_text,
            domain_id=a.get("domain"), epoch=epoch, client=client,
        )["deficit_loose"]
        for a in anchors
    }
    weak = [deficits[a["id"]] for a in anchor_ds.weak(anchors)]
    strong = [deficits[a["id"]] for a in anchor_ds.strong(anchors)]
    return {
        "separation": round(_separation(deficits, anchors), 4),
        "mean_weak_deficit": round(mean(weak), 4) if weak else 0.0,
        "mean_strong_deficit": round(mean(strong), 4) if strong else 0.0,
        "n": len(anchors),
    }


def _latest_frontier() -> dict[str, Any]:
    if not _FRONTIER_DIR.exists():
        return {}
    files = sorted(_FRONTIER_DIR.glob("epoch-*.json"))
    if not files:
        return {}
    try:
        data = json.loads(files[-1].read_text(encoding="utf-8"))
        return data.get("summary", {})
    except Exception:
        return {}


def build_report(client: LemonadeClient | None = None, *, include_agreement: bool = True) -> dict[str, Any]:
    """Full RQGM transparency report for the current champion."""
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    champion_version = versioning.get_champion_version()

    separation = {
        "val": _split_separation(champion_text, anchor_ds.VAL, epoch, client),
        "test": _split_separation(champion_text, anchor_ds.TEST, epoch, client),
    }

    controller = rqgm_adapter.get_controller()
    exploit = controller.assess(
        champion_text, anchor_ds.load_anchors(anchor_ds.VAL), epoch=epoch, client=client, persist=False
    )

    judge_agreement: dict[str, Any] = {}
    if include_agreement:
        judge_agreement = {
            "val": panel.anchor_agreement(champion_text, anchor_ds.load_anchors(anchor_ds.VAL), epoch=epoch, client=client),
            "test": panel.anchor_agreement(champion_text, anchor_ds.load_anchors(anchor_ds.TEST), epoch=epoch, client=client),
        }

    memory: dict[str, Any] = {}
    try:
        from backend.memory import get_memory

        memory = get_memory().stats()
    except Exception as exc:  # pragma: no cover
        memory = {"error": str(exc)}

    return {
        "epoch_id": epoch,
        "champion_version": champion_version,
        "rqgm_backend": rqgm_adapter.RQGM_BACKEND,
        "data_splits": anchor_ds.split_counts(),
        "separation": separation,
        "hack_ratio": exploit.to_dict(),
        "judge_agreement": judge_agreement,
        "frontier": _latest_frontier(),
        "memory": memory,
    }


def health_summary(client: LemonadeClient | None = None) -> dict[str, Any]:
    """Compact, fast evaluator summary for ``/health`` (single-judge, no panel)."""
    epoch = versioning.get_epoch()
    champion_text = versioning.get_champion_rubric_text()
    controller = rqgm_adapter.get_controller()
    exploit = controller.assess(
        champion_text, anchor_ds.load_anchors(anchor_ds.VAL), epoch=epoch, client=client, persist=False
    )
    val_sep = _split_separation(champion_text, anchor_ds.VAL, epoch, client)["separation"]
    test_sep = _split_separation(champion_text, anchor_ds.TEST, epoch, client)["separation"]
    agreement = panel.anchor_agreement(
        champion_text, anchor_ds.load_anchors(anchor_ds.VAL), epoch=epoch, client=client, use_panel=False
    )
    return {
        "rqgm_backend": rqgm_adapter.RQGM_BACKEND,
        "val_separation": val_sep,
        "test_separation": test_sep,
        "hack_ratio": exploit.mean_hack_ratio,
        "exploitation_detected": exploit.exploitation_detected,
        "tolerance_levels": len(exploit.tolerances_after),
        "val_judge_accuracy": agreement["accuracy"],
        "val_judge_kappa": agreement["cohen_kappa"],
    }
