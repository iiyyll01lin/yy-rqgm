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

from backend.evaluator import adversarial, anchors as anchor_ds
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


def _over_acceptance(
    rubric_text: str, epoch: int, client: LemonadeClient | None, *, tau: float
) -> dict[str, Any]:
    """Over-acceptance monitor: fraction of gamed/weak adversarial samples the
    current rubric scores as "strong" (deficit < ``tau``).

    Uses the hardened self-play red team INCLUDING out-of-catalog gaming, so the
    metric reflects generalization (does the rubric wave through disguised /
    novel gaming?) rather than id-memorization. A high rate means the judge is
    fooled by disguised poison pills — an over-optimization / reward-hacking risk.
    """
    samples = adversarial.generate_adversarial_samples(
        rubric_text, include_out_of_catalog=True
    )
    per_sample: list[dict[str, Any]] = []
    accepted = 0
    for s in samples:
        deficit = score_candidate(
            anchor_ds.anchor_candidate_text(s), rubric_text,
            domain_id=s.get("domain"), epoch=epoch, client=client,
        )["deficit_loose"]
        is_strong = deficit < tau
        accepted += int(is_strong)
        per_sample.append({
            "id": s["id"],
            "targets": s["targets"],
            "deficit": round(deficit, 4),
            "accepted_as_strong": is_strong,
            "out_of_catalog": s["id"].startswith("adv_ooc_"),
        })
    n = len(samples)
    return {
        "over_acceptance_rate": round(accepted / n, 4) if n else 0.0,
        "accepted_as_strong": accepted,
        "n": n,
        "tau": tau,
        "per_sample": per_sample,
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

    # Over-optimization monitor: proxy(val) − gold(test) separation gap. A large
    # positive gap means the champion looks much sharper on the split evolution
    # optimizes toward (val) than on the untouched gold split (test).
    proxy_val = separation["val"]["separation"]
    gold_test = separation["test"]["separation"]
    over_optimization = {
        "proxy_val_separation": proxy_val,
        "gold_test_separation": gold_test,
        "separation_gap": round(proxy_val - gold_test, 4),
    }
    over_acceptance = _over_acceptance(champion_text, epoch, client, tau=panel.DEFAULT_TAU)

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
        "over_optimization": over_optimization,
        "over_acceptance": over_acceptance,
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
    over_acc = _over_acceptance(champion_text, epoch, client, tau=panel.DEFAULT_TAU)
    return {
        "rqgm_backend": rqgm_adapter.RQGM_BACKEND,
        "val_separation": val_sep,
        "test_separation": test_sep,
        "proxy_gold_separation_gap": round(val_sep - test_sep, 4),
        "over_acceptance_rate": over_acc["over_acceptance_rate"],
        "hack_ratio": exploit.mean_hack_ratio,
        "exploitation_detected": exploit.exploitation_detected,
        "tolerance_levels": len(exploit.tolerances_after),
        "val_judge_accuracy": agreement["accuracy"],
        "val_judge_kappa": agreement["cohen_kappa"],
    }
