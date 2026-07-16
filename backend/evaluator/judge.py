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
from backend.evaluator import surrogate as surrogate_mod
from backend.evaluator import versioning
from backend.inference.lemonade_client import LemonadeClient, MockMarker, get_lemonade_client
from backend.inference.mock_scoring import SEVERITY_WEIGHT
from backend.inference.parsing import extract_json
from backend.memory import MemoryType, get_memory

_VALID_SEVERITY = {"low", "medium", "high", "critical"}

# Scoring modes (RQGM strict-vs-loose, feeds the hack ratio):
#   loose  = penalise only what the CURRENT rubric covers (what the champion
#            actually scores with today).
#   strict = additionally apply the full poison-pill robustness check.
SCORING_LOOSE = "loose"
SCORING_STRICT = "strict"

# ---------------------------------------------------------------------------
# Structured-output contract (LIVE path)
# ---------------------------------------------------------------------------
# The offline mock always returns exact per-criterion penalties. A live model, if
# left unconstrained, tends to omit them — forcing us to APPROXIMATE penalties
# from red-flag severities (see :func:`_criterion_penalties_from_flags`), which
# degrades the frontier's ``sep::<criterion>`` objectives. To avoid that we send
# an explicit JSON-schema / guided-decoding contract on the live path so the model
# returns per-criterion penalties directly. We still validate + fall back
# gracefully (a non-conforming server never breaks the pipeline).
JUDGE_OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "deficit_loose": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "deficit_strict": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        "criterion_penalties": {
            "type": "object",
            "description": "Per-criterion deficit contribution keyed by rubric criterion id.",
            "additionalProperties": {"type": "number", "minimum": 0.0, "maximum": 1.0},
        },
        "red_flags": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "criterion": {"type": "string"},
                    "severity": {"type": "string", "enum": sorted(_VALID_SEVERITY)},
                    "detail": {"type": "string"},
                },
                "required": ["criterion", "severity"],
            },
        },
        "hack_ratio": {"type": ["number", "null"]},
        "reasoning": {"type": "string"},
    },
    "required": ["deficit_loose", "criterion_penalties", "red_flags"],
}


def judge_response_format() -> dict[str, Any]:
    """OpenAI-compatible ``response_format`` enforcing :data:`JUDGE_OUTPUT_SCHEMA`.

    vLLM honours this via guided decoding; Lemonade / other servers that ignore it
    still work because :func:`evaluate_architecture` validates and falls back.
    """
    return {
        "type": "json_schema",
        "json_schema": {"name": "rqgm_judge_verdict", "schema": JUDGE_OUTPUT_SCHEMA, "strict": False},
    }


def _coerce_penalties(raw_pen: Any) -> dict[str, float] | None:
    """Validate a structured ``criterion_penalties`` object.

    Returns a clean ``{criterion_id: float}`` map, or ``None`` if the payload is
    missing/malformed so the caller can fall back to flag-derived penalties.
    """
    if not isinstance(raw_pen, dict) or not raw_pen:
        return None
    out: dict[str, float] = {}
    for k, v in raw_pen.items():
        try:
            out[str(k)] = float(v)
        except (TypeError, ValueError):
            return None
    return out


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
    # RQGM strict/loose scoring + per-criterion decomposition (Phase 1).
    deficit_loose: float = 0.0
    deficit_strict: float = 0.0
    hack_ratio: float | None = None
    criterion_penalties: dict[str, float] = field(default_factory=dict)
    # Physics-surrogate verdict (advisory; does NOT change the deficit score).
    surrogate: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "deficit_score": self.deficit_score,
            "red_flags": [rf.to_dict() for rf in self.red_flags],
            "reasoning": self.reasoning,
            "epoch_id": self.epoch_id,
            "surrogate": self.surrogate,
        }


def build_rubric_prompt(
    architecture: str,
    domain_id: str | None,
    epoch: int,
    rubric_text: str,
    *,
    scoring_mode: str = SCORING_LOOSE,
    memory_block: str | None = None,
    persona: str | None = None,
) -> list[dict[str, str]]:
    """Compose the (system, user) messages for the judge, with domain merge.

    ``scoring_mode`` selects RQGM strict-vs-loose scoring: ``loose`` applies only
    the base/merged rubric criteria; ``strict`` additionally injects the poison
    pills so the judge must survive them (this is what makes the hack ratio
    meaningful). ``memory_block`` (Phase 3) injects hybrid-search recall.
    ``persona`` (Phase 4 panel) selects a judge persona/temperature for
    self-consistency.
    """
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

    persona_line = f"PANEL_PERSONA: {persona}\n" if persona else ""
    system = (
        f"{MockMarker.EVALUATOR.value}\n"
        f"ACTIVE_EPOCH: {epoch}\n"
        f"SCORING_MODE: {scoring_mode}\n"
        f"{persona_line}"
        "You are the RQGM Evaluator. Apply the champion rubric below to the proposed\n"
        "architecture. Score DEFICITS (0.0 flawless .. 1.0 unacceptable). Think first, then\n"
        "output STRICT JSON per the output_contract.\n\n"
        "=== CHAMPION RUBRIC ===\n"
        f"{rubric_text}\n"
    )
    if domain_block:
        system += f"\n=== DOMAIN CRITERIA (merged) ===\n{domain_block}\n"
    if memory_block:
        system += f"\n=== MEMORY (injected: hybrid_search) ===\n{memory_block}\n"
    if pills_block and scoring_mode == SCORING_STRICT:
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


def retrieve_memory_block(
    query: str,
    epoch: int,
    memory: Any = None,
    *,
    top_k: int = 3,
) -> str:
    """Hybrid-search recall for the judge prompt (implements docs' ``<memory
    injected="hybrid_search">``).

    Pulls active ``heuristic_failure`` memories from this epoch or earlier (so the
    judge never sees future-epoch lessons) plus timeless ``physics_truth`` facts,
    ranked by semantic similarity to the candidate. Activates the previously-dead
    ``EvolutionaryMemory.search`` code path.
    """
    if memory is None:
        try:
            memory = get_memory()
        except Exception:
            return ""
    lines: list[str] = []
    try:
        for h in memory.search(
            query, top_k=top_k, memory_type=MemoryType.HEURISTIC_FAILURE,
            active_only=True, max_epoch=epoch,
        ):
            lines.append(f"- [heuristic_failure @e{h.created_at_epoch}] {h.text}")
        for p in memory.search(
            query, top_k=top_k, memory_type=MemoryType.PHYSICS_TRUTH, active_only=True,
        ):
            lines.append(f"- [physics_truth] {p.text}")
    except Exception:
        return ""
    return "\n".join(lines)


def _clamp(x: float) -> float:
    return max(0.0, min(1.0, x))


def _criterion_penalties_from_flags(red_flags: list[RedFlag]) -> dict[str, float]:
    """Approximate per-criterion deficit contribution from red-flag severities.

    Used on the live-model path (the mock returns exact ``criterion_penalties``).
    """
    out: dict[str, float] = {}
    for rf in red_flags:
        out[rf.criterion] = out.get(rf.criterion, 0.0) + SEVERITY_WEIGHT.get(rf.severity, 0.3)
    return out


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
    *,
    inject_memory: bool = True,
    memory: Any = None,
    seed: int | None = None,
    structured: bool | None = None,
) -> Evaluation:
    """Score ``architecture`` against the champion rubric for ``domain_id``.

    Robust to malformed model output: on a parse failure it degrades to a
    neutral 0.5 deficit with a ``parse_error`` red flag rather than raising.
    When ``inject_memory`` is set, relevant heuristic_failure + physics_truth
    memories are hybrid-search retrieved and injected into the judge prompt.

    ``seed`` threads a deterministic per-request seed to the model (reproducible
    live sampling without collapsing temperature). ``structured`` toggles the
    JSON-schema output contract; when ``None`` (default) it is enabled
    automatically on the live path and disabled for the offline mock (which
    already returns exact per-criterion penalties).
    """
    client = client or get_lemonade_client()
    epoch = versioning.get_epoch() if epoch is None else epoch
    version = versioning.get_champion_version()
    rubric_text = rubric_text if rubric_text is not None else versioning.get_champion_rubric_text()

    use_structured = structured if structured is not None else (not client.using_mock)
    response_format = judge_response_format() if use_structured else None

    # Physics surrogate: predict whether the root-cause variable stays out of bounds
    # after the proposed action (validates the numerical-duct-tape failure mode).
    # Deterministic + offline; advisory (never changes the deficit score).
    surrogate_verdict = surrogate_mod.validate_architecture(architecture, domain_id).to_dict()

    memory_block = retrieve_memory_block(architecture, epoch, memory) if inject_memory else None
    messages = build_rubric_prompt(
        architecture, domain_id, epoch, rubric_text, scoring_mode=SCORING_LOOSE, memory_block=memory_block
    )
    raw = client.chat(
        messages, temperature=0.1, max_tokens=900, seed=seed, response_format=response_format
    )

    parsed = extract_json(raw)
    if parsed is None:
        return Evaluation(
            deficit_score=0.5,
            deficit_loose=0.5,
            deficit_strict=0.5,
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
            surrogate=surrogate_verdict,
        )

    try:
        loose = float(parsed.get("deficit_loose", parsed.get("deficit_score", 0.5)))
    except (TypeError, ValueError):
        loose = 0.5
    loose = _clamp(loose)
    try:
        strict = _clamp(float(parsed.get("deficit_strict", loose)))
    except (TypeError, ValueError):
        strict = loose
    red_flags = _coerce_red_flags(parsed.get("red_flags"))
    # Prefer the model's explicit per-criterion penalties (the structured contract
    # asks for them). Only APPROXIMATE from red-flag severities if the payload is
    # missing/malformed — otherwise the frontier's sep::<criterion> objectives degrade.
    crit_pen = _coerce_penalties(parsed.get("criterion_penalties"))
    if crit_pen is None:
        crit_pen = _criterion_penalties_from_flags(red_flags)
    quality_loose = 1.0 - loose
    hack_ratio = parsed.get("hack_ratio")
    if not isinstance(hack_ratio, (int, float)):
        hack_ratio = (1.0 - strict) / quality_loose if quality_loose > 0 else None

    # Physically CORROBORATE a duct-tape verdict: if the surrogate predicts the
    # root-cause variable stays out of bounds after a masking action, add a
    # physics_common_sense red flag (appended AFTER criterion_penalties are fixed,
    # so the frontier's sep::<criterion> objectives are unaffected).
    if surrogate_verdict.get("is_duct_tape") and surrogate_verdict.get("root_cause_out_of_bounds"):
        if not any(rf.criterion == "physics_common_sense" for rf in red_flags):
            red_flags = red_flags + [
                RedFlag(criterion="physics_common_sense", severity="high",
                        detail=surrogate_verdict.get("detail", "surrogate: root cause out of bounds")),
            ]

    return Evaluation(
        deficit_score=loose,
        deficit_loose=loose,
        deficit_strict=strict,
        hack_ratio=hack_ratio,
        criterion_penalties={str(k): float(v) for k, v in crit_pen.items()},
        red_flags=red_flags,
        reasoning=str(parsed.get("reasoning", parsed.get("thinking", ""))),
        epoch_id=epoch,
        rubric_version=version,
        used_mock=client.using_mock,
        raw=raw,
        surrogate=surrogate_verdict,
    )


def score_candidate(
    candidate_text: str,
    rubric_text: str,
    domain_id: str | None = None,
    epoch: int = 0,
    client: LemonadeClient | None = None,
    *,
    seed: int | None = None,
    structured: bool | None = None,
) -> dict[str, Any]:
    """Score one candidate under ``rubric_text`` in BOTH loose and strict modes.

    Returns ``{deficit_loose, deficit_strict, hack_ratio, red_flags,
    criterion_penalties}``. The offline mock returns both deficits from a single
    call; a live model gets a second (strict) call so the poison pills are
    actually applied — this is what keeps strict != loose (and ``hack_ratio`` off
    a trivial ~1.0) on the live path, not just offline.

    ``seed``/``structured`` mirror :func:`evaluate_architecture`: a reproducible
    per-request seed and the JSON-schema contract (auto-enabled on the live path)
    so the live model returns explicit per-criterion penalties.
    """
    client = client or get_lemonade_client()
    use_structured = structured if structured is not None else (not client.using_mock)
    response_format = judge_response_format() if use_structured else None

    loose_msgs = build_rubric_prompt(candidate_text, domain_id, epoch, rubric_text, scoring_mode=SCORING_LOOSE)
    loose_parsed = extract_json(
        client.chat(loose_msgs, temperature=0.1, max_tokens=900, seed=seed, response_format=response_format)
    ) or {}

    def _f(d: dict, *keys: str, default: float = 0.5) -> float:
        for k in keys:
            if k in d:
                try:
                    return _clamp(float(d[k]))
                except (TypeError, ValueError):
                    continue
        return default

    deficit_loose = _f(loose_parsed, "deficit_loose", "deficit_score")
    red_flags = _coerce_red_flags(loose_parsed.get("red_flags"))
    crit_pen = _coerce_penalties(loose_parsed.get("criterion_penalties"))
    if crit_pen is None:
        crit_pen = _criterion_penalties_from_flags(red_flags)

    if "deficit_strict" in loose_parsed:
        deficit_strict = _f(loose_parsed, "deficit_strict", default=deficit_loose)
    else:
        strict_msgs = build_rubric_prompt(
            candidate_text, domain_id, epoch, rubric_text, scoring_mode=SCORING_STRICT
        )
        strict_parsed = extract_json(
            client.chat(strict_msgs, temperature=0.1, max_tokens=900, seed=seed, response_format=response_format)
        ) or {}
        deficit_strict = _f(strict_parsed, "deficit_strict", "deficit_score", default=deficit_loose)

    quality_loose = 1.0 - deficit_loose
    hack_ratio = (1.0 - deficit_strict) / quality_loose if quality_loose > 0 else None
    return {
        "deficit_loose": deficit_loose,
        "deficit_strict": deficit_strict,
        "hack_ratio": hack_ratio,
        "red_flags": [rf.to_dict() for rf in red_flags],
        "criterion_penalties": {str(k): float(v) for k, v in crit_pen.items()},
    }
