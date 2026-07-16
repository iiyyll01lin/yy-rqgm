"""RQGM transparency report: val/test separation, hack ratio, judge agreement.

Backing data for the ``GET /api/admin/report`` endpoint and the compact summary
added to ``/health``. Everything is computed on held-out anchors under the
current champion so the frontend / docs can show honest, reproducible numbers.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from statistics import mean
from typing import Any

from backend.evaluator import adversarial, anchors as anchor_ds
from backend.evaluator import panel, rqgm_adapter, versioning
from backend.evaluator.gate import _separation
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import (
    DEFAULT_MODEL,
    LemonadeClient,
    get_lemonade_client,
)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_FRONTIER_DIR = _REPO_ROOT / "data" / "frontier"
# Cross-epoch metrics ledger (append-only JSONL time-series). Runtime artifact,
# gitignored — see record_metrics_snapshot / regression_violations below.
_METRICS_DIR = _REPO_ROOT / "data" / "metrics"
METRICS_LEDGER_PATH = _METRICS_DIR / "ledger.jsonl"


def _git_sha() -> str | None:
    """Best-effort current git commit SHA (offline, no subprocess).

    Reads ``.git/HEAD`` and the referenced ref (with a packed-refs fallback) so a
    report can be tied to the exact source revision that produced it. Returns
    ``None`` when repo metadata is unavailable (e.g. running from a built wheel).
    """
    git_dir = _REPO_ROOT / ".git"
    try:
        head = (git_dir / "HEAD").read_text(encoding="utf-8").strip()
    except Exception:
        return None
    if not head.startswith("ref:"):
        return head or None
    ref = head.split(" ", 1)[1].strip()
    try:
        return (git_dir / ref).read_text(encoding="utf-8").strip()
    except Exception:
        pass
    try:
        for line in (git_dir / "packed-refs").read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "^")):
                continue
            sha, _, name = line.partition(" ")
            if name.strip() == ref:
                return sha.strip()
    except Exception:
        return None
    return None


def _provenance(client: LemonadeClient | None) -> dict[str, Any]:
    """Reproducibility provenance: judge model id + git SHA + backend.

    Additive metadata — it records WHICH model and source revision produced the
    numbers so a report is reproducible/auditable. On a live run this is the real
    served model id; offline it is the configured default and ``using_mock`` True.
    """
    resolved = client or get_lemonade_client()
    try:
        judge_model = resolved.model or os.getenv("LEMONADE_MODEL", DEFAULT_MODEL)
    except Exception:
        judge_model = os.getenv("LEMONADE_MODEL", DEFAULT_MODEL)
    try:
        using_mock = bool(resolved.using_mock)
    except Exception:
        using_mock = None
    return {
        "judge_model": judge_model,
        "using_mock": using_mock,
        "rqgm_backend": rqgm_adapter.RQGM_BACKEND,
        "git_sha": _git_sha(),
    }


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
        "provenance": _provenance(client),
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


# ---------------------------------------------------------------------------
# Cross-epoch metrics ledger (regression guard for promotions)
# ---------------------------------------------------------------------------
# An append-only JSONL time-series of the champion's held-out quality metrics,
# one row per snapshot (recorded on every applied promotion; see
# evolve.approve_challenger). It lets us AUDIT that evolution is monotone where it
# must be: a promotion must never reduce the untouched gold ``test`` split's
# separation or judge/human agreement (Cohen's κ). ``data/metrics/`` is gitignored.
_LEDGER_KEYS = (
    "recorded_at", "epoch_id", "champion_version",
    "val_separation", "test_separation", "over_optimization_gap",
    "over_acceptance_rate", "hack_ratio",
    "val_accuracy", "val_kappa", "test_accuracy", "test_kappa",
    "judge_model", "using_mock", "git_sha",
)


def _snapshot_from_report(rep: dict[str, Any]) -> dict[str, Any]:
    ja = rep.get("judge_agreement", {}) or {}
    val_ja, test_ja = ja.get("val", {}) or {}, ja.get("test", {}) or {}
    prov = rep.get("provenance", {}) or {}
    sep = rep.get("separation", {}) or {}
    return {
        "recorded_at": int(time.time()),
        "epoch_id": rep.get("epoch_id"),
        "champion_version": rep.get("champion_version"),
        "val_separation": (sep.get("val", {}) or {}).get("separation"),
        "test_separation": (sep.get("test", {}) or {}).get("separation"),
        "over_optimization_gap": (rep.get("over_optimization", {}) or {}).get("separation_gap"),
        "over_acceptance_rate": (rep.get("over_acceptance", {}) or {}).get("over_acceptance_rate"),
        "hack_ratio": (rep.get("hack_ratio", {}) or {}).get("mean_hack_ratio"),
        "val_accuracy": val_ja.get("accuracy"),
        "val_kappa": val_ja.get("cohen_kappa"),
        "test_accuracy": test_ja.get("accuracy"),
        "test_kappa": test_ja.get("cohen_kappa"),
        "judge_model": prov.get("judge_model"),
        "using_mock": prov.get("using_mock"),
        "git_sha": prov.get("git_sha"),
    }


def record_metrics_snapshot(
    client: LemonadeClient | None = None,
    *,
    report_data: dict[str, Any] | None = None,
    path: Path | str | None = None,
) -> dict[str, Any]:
    """Append the current champion's held-out metrics to the ledger; return the row.

    Pass ``report_data`` to reuse an already-built report (avoids recomputation);
    otherwise a fresh :func:`build_report` (with judge agreement) is used.
    """
    rep = report_data if report_data is not None else build_report(client=client, include_agreement=True)
    row = _snapshot_from_report(rep)
    ledger_path = Path(path) if path is not None else METRICS_LEDGER_PATH
    ledger_path.parent.mkdir(parents=True, exist_ok=True)
    with ledger_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return row


def load_metrics_ledger(path: Path | str | None = None) -> list[dict[str, Any]]:
    """Read the append-only metrics ledger (oldest first)."""
    ledger_path = Path(path) if path is not None else METRICS_LEDGER_PATH
    if not ledger_path.exists():
        return []
    rows: list[dict[str, Any]] = []
    for line in ledger_path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except Exception:
            continue
    return rows


def reset_metrics_ledger(path: Path | str | None = None) -> None:
    """Delete the ledger (test helper / new-deployment baseline)."""
    ledger_path = Path(path) if path is not None else METRICS_LEDGER_PATH
    if ledger_path.exists():
        ledger_path.unlink()


def regression_violations(
    prev: dict[str, Any], curr: dict[str, Any], *, tol: float = 1e-9
) -> list[str]:
    """Return the guard violations if ``curr`` regressed vs ``prev``.

    A promotion must NEVER reduce the untouched gold ``test`` split's separation or
    judge/human agreement (κ) — that would mean evolution bought a proxy (``val``)
    gain by degrading real held-out quality (reward hacking / over-optimization).
    Empty list == the promotion is monotone on the guarded metrics.
    """
    violations: list[str] = []
    for key, label in (("test_separation", "test-split separation"), ("test_kappa", "test-split Cohen's κ")):
        pv, cv = prev.get(key), curr.get(key)
        if pv is None or cv is None:
            continue
        if float(cv) < float(pv) - tol:
            violations.append(f"{label} regressed: {float(pv):.4f} -> {float(cv):.4f}")
    return violations
