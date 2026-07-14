"""LLM-as-judge that scores a proposed agent architecture (deficit scoring).

Uses the current champion rubric (``backend/evaluator/rubric.xml`` at epoch 0,
or a promoted challenger) merged with the active domain pack's criteria +
poison pills, and asks the local model (Lemonade, with mock fallback) to return
a deficit score + red flags. Deterministic offline via the mock.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from backend.domains.registry import get_domain
from backend.evaluator import versioning
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.parsing import extract_json

_VALID_SEVERITY = {"low", "medium", "high", "critical"}


@dataclass(frozen=True)
class RedFlag:
    criterion: str
    severity: str
    detail: str

    def to_dict(self) -> dict:
        return {"criterion": self.criterion, "severity": self.severity, "detail": self.detail}


@dataclass
class Evaluation:
    deficit_score: float
    red_flags: list[RedFlag] = field(default_factory=list)
    reasoning: str = ""
    epoch_id: int = 0
    rubric_version: str = "champion-0"
    used_mock: bool = False
    raw: str = ""

    def to_dict(self) -> dict:
        return {
            "deficit_score": self.deficit_score,
            "red_flags": [rf.to_dict() for rf in self.red_flags],
            "reasoning": self.reasoning,
            "epoch_id": self.epoch_id,
        }


def build_rubric_prompt(architecture: str, domain_id: str | None, epoch: int, rubric_text: str) -> list[dict[str, str]]:
    """Compose the (system, user) messages for the judge, with domain merge."""
    domain_block = ""
    pills_block = ""
    pack = get_domain(domain_id) if domain_id else None
    if pack is not None:
        try:
            domain_block = pack.rubric_fragment() or ""
        except Exception:
            domain_block = ""
        try:
            pills = pack.poison_pills()
        except Exception:
            pills = []
        if pills:
            pills_block = "\n".join(f"- {p}" for p in pills)

    system = (
        f"{MockMarker.EVALUATOR.value}\n"
        f"ACTIVE_EPOCH: {epoch}\n"
        "You are the RQGM Evaluator. Apply the champion rubric below to the proposed\n"
        "architecture. Score DEFICITS (0.0 flawless .. 1.0 unacceptable). Think first, then\n"
        "output STRICT JSON per the output_contract.\n\n"
        "=== CHAMPION RUBRIC ===\n"
        f"{rubric_text}\n"
    )
    if domain_block:
        system += f"\n=== DOMAIN CRITERIA (merged) ===\n{domain_block}\n"
    if pills_block:
        system += f"\n=== POISON PILLS (must be survived) ===\n{pills_block}\n"

    user = (
        "Evaluate this proposed agent architecture"
        + (f" for the '{domain_id}' domain" if domain_id else "")
        + ":\n\n"
        + architecture.strip()
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def _coerce_red_flags(raw_flags: Any) -> list[RedFlag]:
    flags: list[RedFlag] = []
    if not isinstance(raw_flags, list):
        return flags
    for rf in raw_flags:
        if not isinstance(rf, dict):
            continue
        severity = str(rf.get("severity", "medium")).lower()
        if severity not in _VALID_SEVERITY:
            severity = "medium"
        flags.append(
            RedFlag(
                criterion=str(rf.get("criterion", "unspecified")),
                severity=severity,
                detail=str(rf.get("detail", "")),
            )
        )
    return flags


def evaluate_architecture(
    architecture: str,
    domain_id: str | None = None,
    client: LemonadeClient | None = None,
    rubric_text: str | None = None,
    epoch: int | None = None,
) -> Evaluation:
    """Score ``architecture`` against the champion rubric for ``domain_id``.

    Robust to malformed model output: on a parse failure it degrades to a
    neutral 0.5 deficit with a ``parse_error`` red flag rather than raising.
    """
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch() if epoch is None else epoch
    version = versioning.get_champion_version()
    rubric_text = rubric_text if rubric_text is not None else versioning.get_champion_rubric_text()

    messages = build_rubric_prompt(architecture, domain_id, epoch, rubric_text)
    raw = client.chat(messages, temperature=0.1, max_tokens=900)

    parsed = extract_json(raw)
    if parsed is None:
        return Evaluation(
            deficit_score=0.5,
            red_flags=[
                RedFlag(
                    criterion="parse_error",
                    severity="medium",
                    detail="Judge output was not valid JSON; defaulted to neutral deficit.",
                )
            ],
            reasoning="Could not parse evaluator output.",
            epoch_id=epoch,
            rubric_version=version,
            used_mock=client.using_mock,
            raw=raw,
        )

    try:
        score = float(parsed.get("deficit_score", 0.5))
    except (TypeError, ValueError):
        score = 0.5
    score = max(0.0, min(1.0, score))

    return Evaluation(
        deficit_score=score,
        red_flags=_coerce_red_flags(parsed.get("red_flags")),
        reasoning=str(parsed.get("reasoning", parsed.get("thinking", ""))),
        epoch_id=epoch,
        rubric_version=version,
        used_mock=client.using_mock,
        raw=raw,
    )
