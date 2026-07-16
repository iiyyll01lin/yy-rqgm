"""Panel-of-judges + self-consistency aggregation + bias controls (Phase 4).

A single LLM-as-judge is noisy and biased (sycophancy, verbosity/position/self-
preference). This module runs a **panel** of judge personas at different
temperatures, aggregates by self-consistency (median deficit + red-flag vote),
and applies bias controls:

* **criterion-order randomization** — each persona sees the rubric criteria in a
  different (seeded) order, so verdicts cannot depend on ordering;
* **AB/BA position-swap** — with ``position_swap`` each persona scores the
  candidate under BOTH the given criteria order (AB) and the reversed order (BA)
  and the two deficits are averaged, cancelling any residual position/primacy
  bias in a single live judge (offline the deterministic mock is order-invariant,
  so this is a no-op there — behaviour stays deterministic);
* **panel diversity** — multiple personas/temperatures mitigate single-judge
  position/self-preference bias. A per-request ``seed`` keeps each judge
  reproducible WITHOUT collapsing temperature (the panel stays diverse);
* **cross-model judge (wired via config/env)** — a persona may carry its own model
  id (``(name, temperature, model)``) or the whole panel a ``judge_model``; and
  :func:`resolve_panel` routes ONE seat to a different model family when
  ``AGENTFORGE_CROSS_MODEL`` (env) or ``evaluate_panel(cross_model=...)`` is set, so
  a judge less likely to share the primary model's blind spot / self-preference
  scores the same candidate (offline the mock ignores the id → verdict stays
  deterministic; a real different family needs a live model);
* **length normalization** — deficit is a bounded sum of per-criterion penalties
  (not free-form), so a more verbose candidate cannot buy a lower deficit.

The calibrated anchor metric changes from raw separation to **agreement with the
human ground-truth labels** (accuracy + Cohen's κ), which is what actually
matters for a judge.
"""

from __future__ import annotations

import os
import random
import re
from dataclasses import dataclass, field
from statistics import median, pstdev
from typing import Any

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import versioning
from backend.evaluator.judge import (
    SCORING_LOOSE,
    _clamp,
    _coerce_red_flags,
    build_rubric_prompt,
    judge_response_format,
)
from backend.inference.lemonade_client import LemonadeClient, get_lemonade_client
from backend.inference.parsing import extract_json

# (persona name, sampling temperature)
DEFAULT_PANEL: list[tuple[str, float]] = [
    ("principal_reliability_engineer", 0.1),
    ("safety_auditor", 0.3),
    ("adversarial_red_teamer", 0.6),
]

DEFAULT_TAU = 0.3  # deficit >= tau => predicted "weak"

# Cross-model judge config: route ONE panel seat to a DIFFERENT model family so the
# panel does not share a single model's blind spots / self-preference bias. Both
# are env-driven so a live deployment opts in without code changes; offline the
# deterministic mock ignores the model id, so the verdict stays reproducible.
ENV_CROSS_MODEL = "AGENTFORGE_CROSS_MODEL"   # model id for the different-family seat
ENV_PANEL_MODEL = "AGENTFORGE_PANEL_MODEL"   # (optional) model for the rest of the panel


def resolve_panel(
    personas: list[tuple] | None = None, *, cross_model: str | None = None
) -> list[tuple]:
    """Return the panel, pinning the LAST seat to a different-family model if configured.

    A single LLM-as-judge (and a same-family panel) shares self-preference and
    blind spots. When ``cross_model`` (arg) or ``AGENTFORGE_CROSS_MODEL`` (env) is
    set, the last persona is routed to that model — a genuinely different family is
    less likely to endorse the same gamed answer. The other seats keep the panel
    default model. A no-op when unset (same-family panel = the prior behaviour).
    """
    base = [tuple(p) for p in (personas if personas is not None else DEFAULT_PANEL)]
    cm = cross_model if cross_model is not None else os.getenv(ENV_CROSS_MODEL)
    if not cm or not base:
        return base
    name, temp = str(base[-1][0]), float(base[-1][1])
    return base[:-1] + [(name, temp, cm)]

_CRITERION_BLOCK_RE = re.compile(r"[ \t]*<criterion\b.*?</criterion>\s*", re.DOTALL)


@dataclass
class PanelEvaluation:
    deficit_median: float
    deficit_mean: float
    deficit_std: float
    verdict: str
    consensus_red_flags: list[str]
    per_persona: list[dict[str, Any]] = field(default_factory=list)
    n_judges: int = 0
    debate_triggered: bool = False   # P2: a debate round fired (high disagreement)
    debate_rounds: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "deficit_median": round(self.deficit_median, 4),
            "deficit_mean": round(self.deficit_mean, 4),
            "deficit_std": round(self.deficit_std, 4),
            "verdict": self.verdict,
            "consensus_red_flags": self.consensus_red_flags,
            "n_judges": self.n_judges,
            "debate_triggered": self.debate_triggered,
            "debate_rounds": self.debate_rounds,
            "per_persona": self.per_persona,
        }


DEFAULT_DEBATE_THRESHOLD = 0.15  # panel deficit_std >= this => trigger a debate round


def _debate_block(per_persona: list[dict[str, Any]], deficits: list[float]) -> str:
    """Rebuttal context injected into round 2: the spread + each peer's verdict.

    Offline the deterministic mock IGNORES this block (flaws are read only from the
    candidate/user turn), so the debate round is a reproducible no-op; on a live
    model it lets each judge revise after seeing where the panel disagrees.
    """
    med = round(median(deficits), 4) if deficits else 0.0
    lines = [
        f"- {p['persona']} scored deficit {p['deficit']} "
        f"(flags: {', '.join(p['red_flags']) or 'none'})"
        for p in per_persona
    ]
    return (
        "PANEL_DEBATE (high disagreement): the panel median is "
        f"{med} but judges disagree. Reconsider your score against the physical evidence "
        "and your peers below; concede or defend with a concrete mechanism:\n" + "\n".join(lines)
    )


def _reorder_criteria(rubric_text: str, ordered_blocks: list[str], original_blocks: list[str]) -> str:
    """Splice ``ordered_blocks`` back into ``rubric_text`` in place of the originals."""
    out = rubric_text
    for b in original_blocks:
        out = out.replace(b, "\u0000CRIT\u0000", 1)
    for b in ordered_blocks:
        out = out.replace("\u0000CRIT\u0000", b, 1)
    return out


def _shuffle_criteria(rubric_text: str, seed: int) -> str:
    """Bias control: randomize the order criteria appear in (seeded, reversible)."""
    blocks = _CRITERION_BLOCK_RE.findall(rubric_text)
    if len(blocks) < 2:
        return rubric_text
    rng = random.Random(seed)
    shuffled = blocks[:]
    rng.shuffle(shuffled)
    return _reorder_criteria(rubric_text, shuffled, blocks)


def _reverse_criteria(rubric_text: str) -> str:
    """Bias control: present the criteria in reversed order (the 'BA' of AB/BA)."""
    blocks = _CRITERION_BLOCK_RE.findall(rubric_text)
    if len(blocks) < 2:
        return rubric_text
    return _reorder_criteria(rubric_text, list(reversed(blocks)), blocks)


def _persona_spec(spec: Any, default_model: str | None) -> tuple[str, float, str | None]:
    """Unpack a persona entry into ``(name, temperature, model)``.

    A persona is ``(name, temperature)`` or, to enable a CROSS-MODEL panel (a
    config option: different judges are less likely to share the same blind
    spot), ``(name, temperature, model)``. ``default_model`` (the panel-level
    ``judge_model``) is used when a persona does not pin its own.
    """
    if len(spec) >= 3:
        return str(spec[0]), float(spec[1]), (spec[2] or default_model)
    return str(spec[0]), float(spec[1]), default_model


def evaluate_panel(
    architecture: str,
    rubric_text: str,
    *,
    domain_id: str | None = None,
    epoch: int = 0,
    client: LemonadeClient | None = None,
    personas: list[tuple] | None = None,
    randomize_order: bool = True,
    tau: float = DEFAULT_TAU,
    memory_block: str | None = None,
    seed: int | None = None,
    position_swap: bool = False,
    judge_model: str | None = None,
    cross_model: str | None = None,
    debate: bool = False,
    debate_threshold: float = DEFAULT_DEBATE_THRESHOLD,
    structured: bool | None = None,
) -> PanelEvaluation:
    """Run the judge panel on one candidate and aggregate by self-consistency.

    ``seed`` seeds each persona reproducibly (persona ``i`` uses ``seed + i``)
    without collapsing its temperature, so a fixed seed makes a live panel
    reproducible while KEEPING persona/temperature diversity. ``position_swap``
    enables AB/BA order-swap debiasing (see the module docstring). ``judge_model``
    overrides the model for every persona (a persona may still pin its own via a
    ``(name, temperature, model)`` tuple — the cross-model-judge config option).
    """
    client = client or get_lemonade_client()
    if judge_model is None:
        judge_model = os.getenv(ENV_PANEL_MODEL)
    # Route one seat to a different model family if configured (self-preference debias).
    personas = resolve_panel(personas, cross_model=cross_model)
    use_structured = structured if structured is not None else (not client.using_mock)
    response_format = judge_response_format() if use_structured else None

    def _panel_pass(debate_block: str | None, seed_offset: int):
        pp: list[dict[str, Any]] = []
        defs: list[float] = []
        votes: dict[str, int] = {}
        for i, spec in enumerate(personas):
            name, temp, pmodel = _persona_spec(spec, judge_model)
            base_rt = _shuffle_criteria(rubric_text, seed=i) if randomize_order else rubric_text
            # AB/BA position swap: score under the given order AND the reversed order.
            orientations = [base_rt] + ([_reverse_criteria(base_rt)] if position_swap else [])
            persona_seed = None if seed is None else seed + i + seed_offset
            block = memory_block
            if debate_block:
                block = f"{memory_block}\n{debate_block}" if memory_block else debate_block

            run_deficits: list[float] = []
            flags_seen: set[str] = set()
            for j, rt in enumerate(orientations):
                msgs = build_rubric_prompt(
                    architecture, domain_id, epoch, rt,
                    scoring_mode=SCORING_LOOSE, memory_block=block, persona=f"{name}@T={temp}",
                )
                # Distinct (still deterministic) seed per orientation so AB and BA are
                # not identical samples on the live path.
                run_seed = None if persona_seed is None else persona_seed + 10_000 * j
                parsed = extract_json(
                    client.chat(
                        msgs, model=pmodel, temperature=temp, max_tokens=900,
                        seed=run_seed, response_format=response_format,
                    )
                ) or {}
                run_deficits.append(_clamp(float(parsed.get("deficit_loose", parsed.get("deficit_score", 0.5)))))
                for rf in _coerce_red_flags(parsed.get("red_flags")):
                    flags_seen.add(rf.criterion)

            d = sum(run_deficits) / len(run_deficits)
            defs.append(d)
            for cid in flags_seen:
                votes[cid] = votes.get(cid, 0) + 1
            pp.append({"persona": name, "temperature": temp, "deficit": round(d, 4),
                       "model": pmodel, "red_flags": sorted(flags_seen),
                       "orientations": len(orientations)})
        return pp, defs, votes

    # Round 1 — independent scoring.
    per_persona, deficits, flag_votes = _panel_pass(None, 0)

    # P2 debate: only on HIGH disagreement, and NEVER on the promotion gate (which
    # uses score_candidate, not the panel). Offline the mock ignores the injected
    # rebuttal, so this is a reproducible no-op; live, judges may revise.
    debate_triggered = False
    debate_rounds = 0
    if debate and len(deficits) > 1 and pstdev(deficits) >= debate_threshold:
        per_persona, deficits, flag_votes = _panel_pass(
            _debate_block(per_persona, deficits), seed_offset=100_000
        )
        debate_triggered = True
        debate_rounds = 1

    med = median(deficits) if deficits else 0.5
    mean = sum(deficits) / len(deficits) if deficits else 0.5
    std = pstdev(deficits) if len(deficits) > 1 else 0.0
    majority = (len(personas) // 2) + 1
    consensus = sorted(c for c, v in flag_votes.items() if v >= majority)
    verdict = "weak" if med >= tau else "strong"
    return PanelEvaluation(
        deficit_median=med, deficit_mean=mean, deficit_std=std, verdict=verdict,
        consensus_red_flags=consensus, per_persona=per_persona, n_judges=len(personas),
        debate_triggered=debate_triggered, debate_rounds=debate_rounds,
    )


# ---------------------------------------------------------------------------
# Calibrated anchor metric: agreement with human labels (accuracy + Cohen's κ)
# ---------------------------------------------------------------------------
def cohen_kappa(predicted: list[str], truth: list[str]) -> float:
    n = len(predicted)
    if n == 0:
        return 0.0
    po = sum(p == t for p, t in zip(predicted, truth)) / n
    labels = set(predicted) | set(truth)
    pe = sum((predicted.count(x) / n) * (truth.count(x) / n) for x in labels)
    if pe >= 1.0:
        return 1.0
    return (po - pe) / (1.0 - pe)


def anchor_agreement(
    rubric_text: str,
    anchors: list[dict[str, Any]],
    *,
    domain_id: str | None = "smart_manufacturing",
    epoch: int = 0,
    client: LemonadeClient | None = None,
    tau: float = DEFAULT_TAU,
    use_panel: bool = True,
    personas: list[tuple[str, float]] | None = None,
    cross_model: str | None = None,
) -> dict[str, Any]:
    """Judge/human agreement over ``anchors`` (accuracy + Cohen's κ + confusion)."""
    client = client or get_lemonade_client()
    predicted: list[str] = []
    truth: list[str] = []
    per_anchor: list[dict[str, Any]] = []
    for a in anchors:
        cand = anchor_ds.anchor_candidate_text(a)
        if use_panel:
            pe = evaluate_panel(
                cand, rubric_text, domain_id=a.get("domain"), epoch=epoch,
                client=client, personas=personas, tau=tau, cross_model=cross_model,
            )
            deficit = pe.deficit_median
        else:
            from backend.evaluator.judge import score_candidate

            deficit = score_candidate(cand, rubric_text, domain_id=a.get("domain"), epoch=epoch, client=client)["deficit_loose"]
        pred = "weak" if deficit >= tau else "strong"
        gt = a.get("label", "weak")
        predicted.append(pred)
        truth.append(gt)
        per_anchor.append({"id": a["id"], "deficit": round(deficit, 4), "predicted": pred, "label": gt})

    n = len(anchors)
    correct = sum(p == t for p, t in zip(predicted, truth))
    accuracy = correct / n if n else 0.0
    return {
        "n": n,
        "tau": tau,
        "accuracy": round(accuracy, 4),
        "cohen_kappa": round(cohen_kappa(predicted, truth), 4),
        "correct": correct,
        "per_anchor": per_anchor,
    }


def panel_champion_agreement(split: str, *, client: LemonadeClient | None = None) -> dict[str, Any]:
    """Convenience: current champion's panel agreement on a data split."""
    rubric = versioning.get_champion_rubric_text()
    epoch = versioning.get_epoch()
    return anchor_agreement(rubric, anchor_ds.load_anchors(split), epoch=epoch, client=client)
