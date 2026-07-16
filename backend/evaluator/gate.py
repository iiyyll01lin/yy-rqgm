"""RQGM code gate: statistical champion-vs-challenger promotion test.

This is the CODE gate that must pass **before** any human (HITL) is consulted —
the anti-reward-hacking core the shipped code was missing (it promoted a strictly
worse challenger, ``separation_delta = -0.3405``, purely on the HITL boolean).

Gate rules (held-out ``val`` anchors only):

* **P1 — non-inferiority.** ``challenger_sep >= champion_sep`` (a tie favours the
  incumbent, so a challenger must be *strictly* better to advance).
* **P2 — paired bootstrap (BBε-style lower bound).** Resample the val anchors
  (stratified, paired) ``B`` times; require the lower bound of the
  ``(challenger − champion)`` separation confidence interval to be ``> 0``.

Separation = ``mean deficit(weak) − mean deficit(strong)`` under each rubric
(loose scoring = what that rubric actually penalises today). Higher = the rubric
discriminates good from bad architectures better.
"""

from __future__ import annotations

import random
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient


@dataclass
class GateConfig:
    """Tunable strictness for the code gate (defaults sized for a small dataset)."""

    require_non_inferiority: bool = True          # P1
    require_bootstrap: bool = True                # P2
    tie_favors_incumbent: bool = True
    bootstrap_iters: int = 2000
    bootstrap_alpha: float = 0.1                  # one-sided lower bound percentile
    min_separation_margin: float = 0.0            # P1 slack (0 => strict non-inferiority)
    seed: int = 1234


@dataclass
class GateResult:
    passed: bool
    reason: str
    champion_separation: float
    challenger_separation: float
    separation_delta: float
    p1_non_inferior: bool
    p2_passed: bool
    bootstrap_lower_bound: float
    bootstrap_alpha: float
    n_val: int
    champion_deficits: dict[str, float] = field(default_factory=dict)
    challenger_deficits: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _separation(deficits: dict[str, float], anchors: list[dict[str, Any]]) -> float:
    weak = [deficits[a["id"]] for a in anchors if a.get("label") == "weak" and a["id"] in deficits]
    strong = [deficits[a["id"]] for a in anchors if a.get("label") == "strong" and a["id"] in deficits]
    w = mean(weak) if weak else 0.0
    s = mean(strong) if strong else 0.0
    return w - s


def _score_split(
    rubric_text: str,
    anchors: list[dict[str, Any]],
    epoch: int,
    client: LemonadeClient | None,
) -> dict[str, float]:
    out: dict[str, float] = {}
    for a in anchors:
        res = score_candidate(
            anchor_ds.anchor_candidate_text(a),
            rubric_text,
            domain_id=a.get("domain"),
            epoch=epoch,
            client=client,
        )
        out[a["id"]] = res["deficit_loose"]
    return out


def _paired_bootstrap_lower_bound(
    champion: dict[str, float],
    challenger: dict[str, float],
    anchors: list[dict[str, Any]],
    *,
    iters: int,
    alpha: float,
    seed: int,
) -> float:
    """Lower bound of the ``(challenger − champion)`` separation CI.

    Stratified + paired: weak and strong anchors are resampled separately (so a
    resample never collapses one stratum), and the SAME resample is scored for
    both rubrics (paired) to cancel per-anchor variance.
    """
    weak = [a["id"] for a in anchors if a.get("label") == "weak"]
    strong = [a["id"] for a in anchors if a.get("label") == "strong"]
    if not weak or not strong:
        return 0.0
    rng = random.Random(seed)
    deltas: list[float] = []
    for _ in range(iters):
        wi = [rng.choice(weak) for _ in weak]
        si = [rng.choice(strong) for _ in strong]
        champ_sep = mean(champion[i] for i in wi) - mean(champion[j] for j in si)
        chal_sep = mean(challenger[i] for i in wi) - mean(challenger[j] for j in si)
        deltas.append(chal_sep - champ_sep)
    deltas.sort()
    idx = min(len(deltas) - 1, max(0, int(alpha * len(deltas))))
    return deltas[idx]


def separation_lower_bound(
    deficits: dict[str, float],
    anchors: list[dict[str, Any]],
    *,
    iters: int = 1000,
    alpha: float = 0.1,
    seed: int = 7,
) -> float:
    """BBε-style lower CI bound of a single rubric's separation (frontier ranking)."""
    weak = [a["id"] for a in anchors if a.get("label") == "weak"]
    strong = [a["id"] for a in anchors if a.get("label") == "strong"]
    if not weak or not strong:
        return _separation(deficits, anchors)
    rng = random.Random(seed)
    seps: list[float] = []
    for _ in range(iters):
        wi = [rng.choice(weak) for _ in weak]
        si = [rng.choice(strong) for _ in strong]
        seps.append(mean(deficits[i] for i in wi) - mean(deficits[j] for j in si))
    seps.sort()
    idx = min(len(seps) - 1, max(0, int(alpha * len(seps))))
    return seps[idx]


def evaluate_promotion(
    champion_text: str,
    challenger_text: str,
    *,
    domain_id: str | None = None,
    epoch: int = 0,
    val_anchors: list[dict[str, Any]] | None = None,
    config: GateConfig | None = None,
    client: LemonadeClient | None = None,
) -> GateResult:
    """Run the code gate. HITL is only consulted if ``result.passed`` is True."""
    config = config or GateConfig()
    val = val_anchors if val_anchors is not None else anchor_ds.load_anchors(anchor_ds.VAL)

    champ = _score_split(champion_text, val, epoch, client)
    chal = _score_split(challenger_text, val, epoch, client)
    champ_sep = _separation(champ, val)
    chal_sep = _separation(chal, val)
    delta = chal_sep - champ_sep

    if config.tie_favors_incumbent:
        p1 = delta > config.min_separation_margin
    else:
        p1 = delta >= config.min_separation_margin

    lb = _paired_bootstrap_lower_bound(
        champ, chal, val,
        iters=config.bootstrap_iters, alpha=config.bootstrap_alpha, seed=config.seed,
    )
    p2 = lb > 0.0

    checks: list[bool] = []
    if config.require_non_inferiority:
        checks.append(p1)
    if config.require_bootstrap:
        checks.append(p2)
    passed = all(checks) if checks else p1

    if passed:
        reason = (
            f"code gate PASSED: challenger separation {chal_sep:.4f} > champion {champ_sep:.4f} "
            f"(delta {delta:+.4f}); bootstrap lower bound {lb:+.4f} > 0 @ alpha={config.bootstrap_alpha}."
        )
    elif config.require_non_inferiority and not p1:
        reason = (
            f"code gate FAILED (P1 non-inferiority): challenger separation {chal_sep:.4f} "
            f"does not exceed champion {champ_sep:.4f} (delta {delta:+.4f}); tie favours incumbent."
        )
    else:
        reason = (
            f"code gate FAILED (P2 bootstrap): (challenger-champion) separation lower bound "
            f"{lb:+.4f} is not > 0 @ alpha={config.bootstrap_alpha} (delta {delta:+.4f})."
        )

    return GateResult(
        passed=passed,
        reason=reason,
        champion_separation=round(champ_sep, 4),
        challenger_separation=round(chal_sep, 4),
        separation_delta=round(delta, 4),
        p1_non_inferior=p1,
        p2_passed=p2,
        bootstrap_lower_bound=round(lb, 4),
        bootstrap_alpha=config.bootstrap_alpha,
        n_val=len(val),
        champion_deficits={k: round(v, 4) for k, v in champ.items()},
        challenger_deficits={k: round(v, 4) for k, v in chal.items()},
    )
