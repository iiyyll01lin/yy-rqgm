"""Adapter around the official ``rqgm`` package (Red Queen Gödel Machine).

Uses ``rqgm``'s ``EpochManager`` / ``EpochConfig`` / ``TransitionReason`` to:

* compute the **hack ratio** ``mean(quality_strict / quality_loose)`` over a set
  of anchors under the current champion rubric (quality = ``1 − deficit``), and
* trigger ``EXPLOITATION_DETECTED`` + **tolerance tightening** when the rubric has
  a poison-pill blind spot (loose passes what strict fails).

The ``rqgm`` tolerance schedule is mapped to our evaluator's strictness: dropping
the loosest tolerance level == the evaluator gets stricter (RQGM §3.4). The
active schedule is persisted to ``data/rqgm_state.json`` so tightening carries
across epochs.

If ``rqgm`` cannot be imported (e.g. fully air-gapped with no wheel cached), a
faithful local fallback reproduces the same hack-ratio + single-level-drop logic
so the platform still runs; :data:`RQGM_BACKEND` records which path is active.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from statistics import mean
from typing import Any

# ``rqgm`` is a HARD dependency (see pyproject ``dependencies``), so the primary
# branch below is the one that actually runs in every correctly-installed
# environment — it is covered by the test suite (and pinned by
# ``tests/test_live_wiring.py::test_rqgm_backend_is_the_real_package``). The
# ``except`` fallback is the branch that is NEVER covered in that environment
# (it only runs if the wheel is somehow unavailable, e.g. a broken air-gapped
# install), so the ``# pragma: no cover`` belongs on the FALLBACK, not here.
try:  # primary path: the official package (installed from PyPI, offline-cacheable)
    from rqgm import (
        DEFAULT_TOLERANCES,
        MIN_TOLERANCE_LEVELS,
        EpochConfig,
        EpochManager,
        TransitionReason,
        UtilityMutationParams,
    )

    RQGM_BACKEND = "rqgm"
    _RQGM_AVAILABLE = True
except Exception:  # pragma: no cover - fallback only if the rqgm wheel is unavailable
    DEFAULT_TOLERANCES = [0.0, 0.001, 0.01, 0.025, 0.05, 0.1]
    MIN_TOLERANCE_LEVELS = 2
    RQGM_BACKEND = "local-fallback"
    _RQGM_AVAILABLE = False

from backend.evaluator import anchors as anchor_ds
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
_STATE_PATH = _REPO_ROOT / "data" / "rqgm_state.json"

DEFAULT_HACK_RATIO_THRESHOLD = 0.6


@dataclass
class ExploitationReport:
    mean_hack_ratio: float | None
    exploitation_detected: bool
    reason: str
    tolerances_before: list[float]
    tolerances_after: list[float]
    tightened: bool
    trigger_adversarial_injection: bool
    n_samples: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "backend": RQGM_BACKEND,
            "mean_hack_ratio": self.mean_hack_ratio,
            "exploitation_detected": self.exploitation_detected,
            "reason": self.reason,
            "tolerances_before": self.tolerances_before,
            "tolerances_after": self.tolerances_after,
            "tightened": self.tightened,
            "trigger_adversarial_injection": self.trigger_adversarial_injection,
            "strictness_level": len(DEFAULT_TOLERANCES) - len(self.tolerances_after),
            "n_samples": self.n_samples,
        }


def _mean_hack_ratio(strict_q: list[float], loose_q: list[float]) -> float | None:
    ratios = [st / lo for st, lo in zip(strict_q, loose_q) if lo > 0]
    return mean(ratios) if ratios else None


def _local_new_tolerances(current: list[float], is_exploiting: bool, max_drops: int) -> list[float]:
    new = list(current)
    if is_exploiting:
        drops = 0
        while drops < max_drops and len(new) > MIN_TOLERANCE_LEVELS and len(new) > 1:
            new.pop()
            drops += 1
    elif len(new) < len(DEFAULT_TOLERANCES):
        for tol in DEFAULT_TOLERANCES:
            if tol not in new:
                new.append(tol)
                new.sort()
                break
    return new


def detect_exploitation(
    strict_qualities: list[float],
    loose_qualities: list[float],
    current_tolerances: list[float] | None = None,
    *,
    threshold: float = DEFAULT_HACK_RATIO_THRESHOLD,
    max_drops: int = 1,
) -> ExploitationReport:
    """Pure hack-ratio assessment (no persistence).

    ``*_qualities`` are per-sample quality scores (``1 − deficit``) under strict
    and loose scoring. Exploitation fires when the mean ratio dips below
    ``threshold`` — i.e. the rubric systematically passes (loose) what the
    ground-truth poison-pill check fails (strict).
    """
    current = list(current_tolerances) if current_tolerances is not None else list(DEFAULT_TOLERANCES)
    n = min(len(strict_qualities), len(loose_qualities))
    ratio = _mean_hack_ratio(strict_qualities[:n], loose_qualities[:n])

    if _RQGM_AVAILABLE and n > 0:
        cfg = EpochConfig(
            epoch_size=n,
            min_improvement_threshold=0.0,  # isolate the exploitation signal
            exploitation_hack_ratio_threshold=threshold,
            utility_mutation_params=UtilityMutationParams(max_tolerance_drops_per_epoch=max_drops),
        )
        mgr = EpochManager(cfg, initial_tolerances=current)
        for i in range(n):
            mgr.record_iteration_result(i, 1.0, strict_qualities[i], loose_qualities[i])
        transition = mgr.evaluate_epoch_boundary(n - 1)
        new_tol = list(transition.new_tolerances)
        reason = transition.reason.name
        exploitation = transition.reason == TransitionReason.EXPLOITATION_DETECTED
        adversarial = bool(transition.trigger_adversarial_injection)
    else:
        exploitation = ratio is not None and ratio < threshold
        new_tol = _local_new_tolerances(current, exploitation, max_drops)
        reason = "EXPLOITATION_DETECTED" if exploitation else "NO_TRANSITION"
        adversarial = False

    return ExploitationReport(
        mean_hack_ratio=round(ratio, 4) if ratio is not None else None,
        exploitation_detected=exploitation,
        reason=reason,
        tolerances_before=current,
        tolerances_after=new_tol,
        tightened=new_tol != current,
        trigger_adversarial_injection=adversarial,
        n_samples=n,
    )


# ---------------------------------------------------------------------------
# Persistent controller (tolerance schedule survives across epochs)
# ---------------------------------------------------------------------------
class RqgmController:
    """Stateful wrapper that persists the active tolerance schedule."""

    def __init__(self, state_path: Path | None = None, threshold: float = DEFAULT_HACK_RATIO_THRESHOLD):
        self.state_path = state_path or _STATE_PATH
        self.threshold = threshold

    def _load(self) -> dict[str, Any]:
        if self.state_path.exists():
            try:
                return json.loads(self.state_path.read_text(encoding="utf-8"))
            except Exception:
                pass
        return {"tolerances": list(DEFAULT_TOLERANCES), "history": []}

    def _save(self, state: dict[str, Any]) -> None:
        self.state_path.parent.mkdir(parents=True, exist_ok=True)
        self.state_path.write_text(json.dumps(state, indent=2), encoding="utf-8")

    def current_tolerances(self) -> list[float]:
        return list(self._load().get("tolerances", DEFAULT_TOLERANCES))

    def qualities_for_rubric(
        self,
        rubric_text: str,
        anchors: list[dict[str, Any]],
        *,
        epoch: int = 0,
        client: LemonadeClient | None = None,
    ) -> tuple[list[float], list[float]]:
        strict_q: list[float] = []
        loose_q: list[float] = []
        for a in anchors:
            res = score_candidate(
                anchor_ds.anchor_candidate_text(a),
                rubric_text,
                domain_id=a.get("domain"),
                epoch=epoch,
                client=client,
            )
            loose_q.append(1.0 - res["deficit_loose"])
            strict_q.append(1.0 - res["deficit_strict"])
        return strict_q, loose_q

    def assess(
        self,
        rubric_text: str,
        anchors: list[dict[str, Any]] | None = None,
        *,
        epoch: int = 0,
        client: LemonadeClient | None = None,
        persist: bool = True,
        weak_only: bool = True,
    ) -> ExploitationReport:
        """Assess exploitation of ``rubric_text`` on ``anchors`` and (optionally)
        persist any tolerance tightening.

        The hack ratio is measured over the *weak/gamed* anchors by default — the
        population that can actually expose a poison-pill blind spot (strong
        anchors have no poison flaws, so their ratio is trivially 1.0 and would
        only dilute the signal)."""
        anchors = anchors if anchors is not None else anchor_ds.load_anchors(anchor_ds.VAL)
        if weak_only:
            gamed = anchor_ds.weak(anchors)
            anchors = gamed or anchors
        strict_q, loose_q = self.qualities_for_rubric(rubric_text, anchors, epoch=epoch, client=client)
        current = self.current_tolerances()
        report = detect_exploitation(strict_q, loose_q, current, threshold=self.threshold)
        if persist and report.tightened:
            state = self._load()
            state["tolerances"] = report.tolerances_after
            state.setdefault("history", []).append(
                {
                    "epoch": epoch,
                    "reason": report.reason,
                    "hack_ratio": report.mean_hack_ratio,
                    "tolerances": report.tolerances_after,
                }
            )
            self._save(state)
        return report

    def reset(self) -> None:
        if self.state_path.exists():
            self.state_path.unlink()


_controller: RqgmController | None = None


def get_controller() -> RqgmController:
    global _controller
    if _controller is None:
        _controller = RqgmController()
    return _controller
