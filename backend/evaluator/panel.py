"""Panel-of-judges + self-consistency aggregation + bias controls (Phase 4).

A single LLM-as-judge is noisy and biased (sycophancy, verbosity/position/self-
preference). This module runs a **panel** of judge personas at different
temperatures, aggregates by self-consistency (median deficit + red-flag vote),
and applies bias controls:

* **criterion-order randomization** — each persona sees the rubric criteria in a
  different (seeded) order, so verdicts cannot depend on ordering;
* **panel diversity** — multiple personas/temperatures mitigate single-judge
  position/self-preference bias;
* **length normalization** — deficit is a bounded sum of per-criterion penalties
  (not free-form), so a more verbose candidate cannot buy a lower deficit.

The calibrated anchor metric changes from raw separation to **agreement with the
human ground-truth labels** (accuracy + Cohen's κ), which is what actually
matters for a judge.
"""

from __future__ import annotations

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

    def to_dict(self) -> dict[str, Any]:
        return {
            "deficit_median": round(self.deficit_median, 4),
            "deficit_mean": round(self.deficit_mean, 4),
            "deficit_std": round(self.deficit_std, 4),
            "verdict": self.verdict,
            "consensus_red_flags": self.consensus_red_flags,
            "n_judges": self.n_judges,
            "per_persona": self.per_persona,
        }


def _shuffle_criteria(rubric_text: str, seed: int) -> str:
    """Bias control: randomize the order criteria appear in (seeded, reversible)."""
    blocks = _CRITERION_BLOCK_RE.findall(rubric_text)
    if len(blocks) < 2:
        return rubric_text
    rng = random.Random(seed)
    shuffled = blocks[:]
    rng.shuffle(shuffled)
    out = rubric_text
    # Replace the first criterion block with a placeholder run, then re-inject in
    # shuffled order. Simplest robust approach: remove all, then splice back.
    for b in blocks:
        out = out.replace(b, "\u0000CRIT\u0000", 1)
    for b in shuffled:
        out = out.replace("\u0000CRIT\u0000", b, 1)
    return out


def evaluate_panel(
    architecture: str,
    rubric_text: str,
    *,
    domain_id: str | None = None,
    epoch: int = 0,
    client: LemonadeClient | None = None,
    personas: list[tuple[str, float]] | None = None,
    randomize_order: bool = True,
    tau: float = DEFAULT_TAU,
    memory_block: str | None = None,
) -> PanelEvaluation:
    """Run the judge panel on one candidate and aggregate by self-consistency."""
    client = client or get_lemonade_client()
    personas = personas or DEFAULT_PANEL

    per_persona: list[dict[str, Any]] = []
    deficits: list[float] = []
    flag_votes: dict[str, int] = {}
    for i, (name, temp) in enumerate(personas):
        rt = _shuffle_criteria(rubric_text, seed=i) if randomize_order else rubric_text
        msgs = build_rubric_prompt(
            architecture, domain_id, epoch, rt,
            scoring_mode=SCORING_LOOSE, memory_block=memory_block, persona=f"{name}@T={temp}",
        )
        parsed = extract_json(client.chat(msgs, temperature=temp, max_tokens=900)) or {}
        d = _clamp(float(parsed.get("deficit_loose", parsed.get("deficit_score", 0.5))))
        flags = _coerce_red_flags(parsed.get("red_flags"))
        deficits.append(d)
        for cid in {rf.criterion for rf in flags}:
            flag_votes[cid] = flag_votes.get(cid, 0) + 1
        per_persona.append({"persona": name, "temperature": temp, "deficit": round(d, 4),
                            "red_flags": [rf.criterion for rf in flags]})

    med = median(deficits) if deficits else 0.5
    mean = sum(deficits) / len(deficits) if deficits else 0.5
    std = pstdev(deficits) if len(deficits) > 1 else 0.0
    majority = (len(personas) // 2) + 1
    consensus = sorted(c for c, v in flag_votes.items() if v >= majority)
    verdict = "weak" if med >= tau else "strong"
    return PanelEvaluation(
        deficit_median=med, deficit_mean=mean, deficit_std=std, verdict=verdict,
        consensus_red_flags=consensus, per_persona=per_persona, n_judges=len(personas),
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
                client=client, personas=personas, tau=tau,
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
