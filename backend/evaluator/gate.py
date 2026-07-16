"""RQGM code gate: statistical champion-vs-challenger promotion test.

This is the CODE gate that must pass **before** any human (HITL) is consulted —
the anti-reward-hacking core the shipped code was missing (it promoted a strictly
worse challenger, ``separation_delta = -0.3405``, purely on the HITL boolean).

Gate rules (held-out ``val`` anchors only):

* **P1 — non-inferiority.** ``challenger_sep >= champion_sep`` (a tie favours the
  incumbent, so a challenger must be *strictly* better to advance).
* **P2 — Bayesian Beta-Binomial posterior + minimum effect (MDE).** The old P2
  was a paired bootstrap on ~7 val anchors where the strong term cancels offline
  (effective N≈4 weak); it structurally rejected genuine single-flaw-family
  fixes. It is replaced by a **Beta-Binomial posterior over the per-anchor paired
  *win indicators*** on the weak (gamed) anchors: for each weak anchor a "win" is
  ``challenger_deficit > champion_deficit`` (the challenger penalises that held-out
  weak architecture *more*), a "loss" the reverse, ties carry no directional
  evidence. With a ``Beta(prior_alpha + wins, prior_beta + losses)`` posterior on
  the win-rate ``θ`` we report ``P(Δseparation > 0) := P(θ > 0.5)`` (the posterior
  probability the challenger separates the held-out set better than the incumbent).
  A challenger is promoted iff that posterior ``>= posterior_threshold`` (default
  0.95) **and** the observed effect ``Δsep >= min_detectable_effect`` (MDE,
  default 0.10). A per-flaw-family breakdown is reported for transparency.

Separation = ``mean deficit(weak) − mean deficit(strong)`` under each rubric
(loose scoring = what that rubric actually penalises today). Higher = the rubric
discriminates good from bad architectures better. Offline the strong term is
identical for both rubrics, so ``Δsep`` reduces to the mean of the per-anchor
weak deltas — exactly the quantity the Beta-Binomial posterior tests the sign of.
"""

from __future__ import annotations

import json
import math
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from statistics import mean
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator.judge import score_candidate
from backend.inference.lemonade_client import LemonadeClient

_REPO_ROOT = Path(__file__).resolve().parents[2]
# Cross-epoch sequential-gate ledger (anytime-valid family-wise error control).
# Runtime state, gitignored — the immutable champion seed lives in rubric.xml.
SEQUENTIAL_STATE_PATH = _REPO_ROOT / "data" / "gate_sequential.json"


@dataclass
class GateConfig:
    """Tunable strictness for the code gate (defaults sized for a small dataset).

    P2 is a Bayesian Beta-Binomial posterior over per-anchor paired win
    indicators (see the module docstring). ``posterior_threshold`` is the minimum
    ``P(Δsep>0)`` required and ``min_detectable_effect`` the minimum practical
    effect size (MDE) — both must clear for a challenger to advance.
    """

    require_non_inferiority: bool = True          # P1
    require_posterior: bool = True                # P2 (Bayesian)
    tie_favors_incumbent: bool = True
    min_separation_margin: float = 0.0            # P1 slack (0 => strict non-inferiority)
    # P2 — Beta-Binomial posterior over paired per-anchor win indicators.
    posterior_threshold: float = 0.95             # promote iff P(Δsep>0) >= this
    min_detectable_effect: float = 0.10           # MDE: require Δsep >= this
    prior_alpha: float = 1.0                      # Beta prior on wins (Bayes-Laplace)
    prior_beta: float = 1.0                       # Beta prior on losses
    win_epsilon: float = 1e-6                     # |per-anchor delta| below this => tie
    # --- cross-epoch anytime-valid correction (family-wise error control) -----
    # OFF by default so the per-epoch P1+P2 gate is unchanged. When enabled, an
    # e-process (a test martingale of per-epoch e-values, persisted across epochs)
    # additionally guards against family-wise error from REPEATED promotions: the
    # running wealth must clear ``1/family_wise_alpha`` (Ville's inequality), so
    # promoting repeatedly on noise (e-values ≈ 1) can never accumulate enough
    # wealth to keep passing. See :func:`epoch_e_value`.
    sequential_correction: bool = False
    family_wise_alpha: float = 0.05               # global type-I budget over ALL promotions


@dataclass
class GateResult:
    passed: bool
    reason: str
    champion_separation: float
    challenger_separation: float
    separation_delta: float
    p1_non_inferior: bool
    p2_passed: bool
    # P2 (Bayesian) transparency:
    posterior_prob_improvement: float             # P(Δsep>0) = P(θ>0.5)
    posterior_threshold: float
    effect_size: float                            # == separation_delta (the MDE test target)
    min_detectable_effect: float
    n_wins: int
    n_losses: int
    n_ties: int
    n_val: int
    n_weak: int
    # Cross-epoch sequential correction (anytime-valid; advisory unless
    # ``config.sequential_correction`` is set, in which case it also gates ``passed``).
    sequential_correction: bool = False
    sequential_e_value: float = 1.0               # this epoch's e-value (Bayes factor vs θ=0.5)
    sequential_log_wealth: float = 0.0            # log of the PROSPECTIVE running wealth
    sequential_threshold: float = 0.0             # 1 / family_wise_alpha (the wealth bar)
    family_wise_alpha: float = 0.05
    sequential_passed: bool = True                # prospective wealth >= threshold
    sequential_looks_prior: int = 0               # committed promotions before this look
    per_flaw_wins: dict[str, dict[str, int]] = field(default_factory=dict)
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


def _betacf(x: float, a: float, b: float) -> float:
    """Continued fraction for the incomplete beta function (Numerical Recipes)."""
    MAXIT, EPS, FPMIN = 300, 3.0e-14, 1.0e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < FPMIN:
        d = FPMIN
    d = 1.0 / d
    h = d
    for m in range(1, MAXIT + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < FPMIN:
            d = FPMIN
        c = 1.0 + aa / c
        if abs(c) < FPMIN:
            c = FPMIN
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < EPS:
            break
    return h


def regularized_incomplete_beta(x: float, a: float, b: float) -> float:
    """Regularised incomplete beta ``I_x(a, b)`` (== Beta(a, b) CDF at ``x``)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    lbeta = math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)
    front = math.exp(a * math.log(x) + b * math.log(1.0 - x) - lbeta)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(x, a, b) / a
    return 1.0 - front * _betacf(1.0 - x, b, a) / b


def beta_binomial_superiority(
    wins: int, losses: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> float:
    """Posterior ``P(θ > 0.5)`` for a Beta-Binomial win-rate model.

    ``θ`` is the probability a random held-out weak anchor is penalised *more* by
    the challenger than the champion. ``P(θ > 0.5)`` is the posterior probability
    the challenger is the more-separating rubric — reported as ``P(Δsep > 0)``.
    """
    a_post = prior_alpha + wins
    b_post = prior_beta + losses
    return 1.0 - regularized_incomplete_beta(0.5, a_post, b_post)


def _log_beta(a: float, b: float) -> float:
    return math.lgamma(a) + math.lgamma(b) - math.lgamma(a + b)


def epoch_e_value(
    wins: int, losses: int, *, prior_alpha: float = 1.0, prior_beta: float = 1.0
) -> float:
    """Per-epoch **one-sided e-value** for ``H0: θ = 0.5`` vs ``H1: θ > 0.5``.

    A Bayes factor against the simple null ``θ = 0.5`` (challenger no better on the
    per-anchor win indicators) with a proper alternative that is a ``Beta(prior_alpha,
    prior_beta)`` prior TRUNCATED to ``θ ∈ (0.5, 1)`` — i.e. it only rewards evidence
    that the challenger wins *more* than half the time, so all-LOSSES gives ``e < 1``
    (evidence for the null), not a spurious large value::

        m1 = B(a+W, b+L)/B(a, b) · P(θ>0.5 | Beta(a+W, b+L)) / P(θ>0.5 | Beta(a, b))
        e  = m1 / 0.5^(W+L)

    Because ``E_H0[e] = 1``, the PRODUCT of per-epoch e-values across promotions is a
    test (super)martingale, so by Ville's inequality the probability its running
    maximum ever exceeds ``1/α`` is ``≤ α`` — an **anytime-valid** family-wise
    guarantee needing no pre-specified number of looks. Genuine improvements
    (``W ≫ L``) make ``e ≫ 1`` and accumulate wealth; promoting on noise
    (``W ≈ L``) gives ``e ≈ 1`` and the wealth never clears ``1/α``.
    """
    n = wins + losses
    if n <= 0:
        return 1.0
    prior_tail = 1.0 - regularized_incomplete_beta(0.5, prior_alpha, prior_beta)
    post_tail = 1.0 - regularized_incomplete_beta(0.5, prior_alpha + wins, prior_beta + losses)
    if prior_tail <= 0.0 or post_tail <= 0.0:
        return 0.0
    log_e = (
        _log_beta(prior_alpha + wins, prior_beta + losses)
        - _log_beta(prior_alpha, prior_beta)
        + math.log(post_tail)
        - math.log(prior_tail)
        + n * math.log(2.0)  # divide by the null likelihood 0.5^n
    )
    return math.exp(log_e)


# ---------------------------------------------------------------------------
# Sequential-gate ledger (persisted e-process wealth across epochs)
# ---------------------------------------------------------------------------
def load_sequential_state() -> dict[str, Any]:
    """Persisted e-process state: ``{log_wealth, looks, history}`` (wealth=1 fresh)."""
    try:
        return json.loads(SEQUENTIAL_STATE_PATH.read_text(encoding="utf-8"))
    except Exception:
        return {"log_wealth": 0.0, "looks": 0, "history": []}


def _save_sequential_state(state: dict[str, Any]) -> None:
    SEQUENTIAL_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)
    SEQUENTIAL_STATE_PATH.write_text(json.dumps(state, indent=2), encoding="utf-8")


def reset_sequential_state() -> None:
    """Reset the e-process wealth to 1 (test helper / new-deployment baseline)."""
    if SEQUENTIAL_STATE_PATH.exists():
        SEQUENTIAL_STATE_PATH.unlink()


def commit_sequential_look(
    e_value: float, *, epoch: int | None = None, version: str | None = None
) -> dict[str, Any]:
    """Multiply the committed e-process wealth by ``e_value`` (one accepted look).

    Called by the promotion path ONLY when a challenger is actually promoted, so
    the family-wise budget is spent per PROMOTION. Uses log-wealth to avoid
    overflow across many epochs.
    """
    state = load_sequential_state()
    state["log_wealth"] = float(state.get("log_wealth", 0.0)) + math.log(max(e_value, 1e-300))
    state["looks"] = int(state.get("looks", 0)) + 1
    state.setdefault("history", []).append(
        {
            "epoch": epoch,
            "version": version,
            "e_value": round(float(e_value), 6),
            "log_wealth": round(state["log_wealth"], 6),
        }
    )
    _save_sequential_state(state)
    return state


def _paired_win_indicators(
    champion: dict[str, float],
    challenger: dict[str, float],
    weak_anchors: list[dict[str, Any]],
    *,
    epsilon: float,
) -> tuple[int, int, int, dict[str, dict[str, int]]]:
    """Per-anchor paired win/loss/tie counts on the weak anchors.

    A *win* = the challenger assigns a strictly higher deficit than the champion
    to that held-out weak (gamed) architecture (it caught more of the planted
    failure); a *loss* is the reverse; a *tie* carries no directional evidence.
    Also returns a per-flaw-family breakdown (a hierarchical view: which planted
    failure modes the change actually moves) for transparency.
    """
    wins = losses = ties = 0
    per_flaw: dict[str, dict[str, int]] = {}
    for a in weak_anchors:
        aid = a["id"]
        if aid not in champion or aid not in challenger:
            continue
        delta = challenger[aid] - champion[aid]
        if delta > epsilon:
            outcome, wins = "win", wins + 1
        elif delta < -epsilon:
            outcome, losses = "loss", losses + 1
        else:
            outcome, ties = "tie", ties + 1
        for flaw in a.get("flaws", []):
            slot = per_flaw.setdefault(flaw, {"win": 0, "loss": 0, "tie": 0})
            slot[outcome] += 1
    return wins, losses, ties, per_flaw


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

    # P2 — Bayesian Beta-Binomial posterior over paired per-anchor win indicators
    # on the weak (gamed) anchors, plus a minimum practical effect (MDE) on Δsep.
    weak = anchor_ds.weak(val)
    wins, losses, ties, per_flaw = _paired_win_indicators(
        champ, chal, weak, epsilon=config.win_epsilon
    )
    posterior = beta_binomial_superiority(
        wins, losses, prior_alpha=config.prior_alpha, prior_beta=config.prior_beta
    )
    posterior_ok = posterior >= config.posterior_threshold
    effect_ok = delta >= config.min_detectable_effect
    p2 = posterior_ok and effect_ok

    # P3 — cross-epoch anytime-valid correction (e-process). Compute the e-value +
    # PROSPECTIVE running wealth regardless (advisory), but only gate ``passed`` on
    # it when ``config.sequential_correction`` is set. The ledger is read-only here;
    # the promotion path commits the look (see :func:`commit_sequential_look`).
    e_value = epoch_e_value(
        wins, losses, prior_alpha=config.prior_alpha, prior_beta=config.prior_beta
    )
    seq_state = load_sequential_state()
    prior_log_wealth = float(seq_state.get("log_wealth", 0.0))
    prospective_log_wealth = prior_log_wealth + math.log(max(e_value, 1e-300))
    seq_threshold = 1.0 / config.family_wise_alpha if config.family_wise_alpha > 0 else float("inf")
    seq_passed = prospective_log_wealth >= math.log(seq_threshold)

    checks: list[bool] = []
    if config.require_non_inferiority:
        checks.append(p1)
    if config.require_posterior:
        checks.append(p2)
    if config.sequential_correction:
        checks.append(seq_passed)
    passed = all(checks) if checks else p1

    if passed:
        reason = (
            f"code gate PASSED: challenger separation {chal_sep:.4f} > champion {champ_sep:.4f} "
            f"(delta {delta:+.4f}); posterior P(Δsep>0)={posterior:.4f} >= {config.posterior_threshold} "
            f"on {wins}W/{losses}L/{ties}T weak anchors and effect {delta:.4f} >= MDE "
            f"{config.min_detectable_effect}."
            + (
                f" Sequential e-process wealth {math.exp(prospective_log_wealth):.2f} "
                f">= 1/α={seq_threshold:.1f}."
                if config.sequential_correction
                else ""
            )
        )
    elif config.require_non_inferiority and not p1:
        reason = (
            f"code gate FAILED (P1 non-inferiority): challenger separation {chal_sep:.4f} "
            f"does not exceed champion {champ_sep:.4f} (delta {delta:+.4f}); tie favours incumbent."
        )
    elif not posterior_ok:
        reason = (
            f"code gate FAILED (P2 posterior): P(Δsep>0)={posterior:.4f} < {config.posterior_threshold} "
            f"(only {wins}W/{losses}L/{ties}T weak anchors moved; underpowered / inconsistent gain)."
        )
    elif not effect_ok:
        reason = (
            f"code gate FAILED (P2 effect size): posterior P(Δsep>0)={posterior:.4f} is sufficient but "
            f"the effect {delta:.4f} is below the minimum practical size (MDE "
            f"{config.min_detectable_effect})."
        )
    else:
        reason = (
            f"code gate FAILED (P3 sequential): e-process wealth "
            f"{math.exp(prospective_log_wealth):.2f} < 1/α={seq_threshold:.1f} (this epoch's "
            f"e-value {e_value:.2f} on {wins}W/{losses}L is not enough accumulated evidence to "
            f"promote again without inflating family-wise error)."
        )

    return GateResult(
        passed=passed,
        reason=reason,
        champion_separation=round(champ_sep, 4),
        challenger_separation=round(chal_sep, 4),
        separation_delta=round(delta, 4),
        p1_non_inferior=p1,
        p2_passed=p2,
        posterior_prob_improvement=round(posterior, 4),
        posterior_threshold=config.posterior_threshold,
        effect_size=round(delta, 4),
        min_detectable_effect=config.min_detectable_effect,
        n_wins=wins,
        n_losses=losses,
        n_ties=ties,
        n_val=len(val),
        n_weak=len(weak),
        sequential_correction=config.sequential_correction,
        sequential_e_value=round(e_value, 6),
        sequential_log_wealth=round(prospective_log_wealth, 6),
        sequential_threshold=round(seq_threshold, 4),
        family_wise_alpha=config.family_wise_alpha,
        sequential_passed=seq_passed,
        sequential_looks_prior=int(seq_state.get("looks", 0)),
        per_flaw_wins=per_flaw,
        champion_deficits={k: round(v, 4) for k, v in champ.items()},
        challenger_deficits={k: round(v, 4) for k, v in chal.items()},
    )
