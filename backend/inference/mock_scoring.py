"""Rubric-aware deterministic scoring for the offline evaluator mock.

The old mock scored an architecture by a *hash of the prompt*, so the
champion-vs-challenger ``separation`` signal was pure noise offline and the RQGM
code gate would have been meaningless without a live model. This module replaces
that with a deterministic, **rubric-aware** scorer:

* Anchor architectures carry *planted flaw tags* (and strength tags) in a
  machine-readable sentinel (see :func:`format_sentinel`). A flaw only raises the
  deficit if the **current rubric contains a criterion that catches it** — so a
  genuinely better rubric (one that adds coverage for a previously-missed flaw)
  measurably increases the weak-vs-strong separation.
* Two scores are produced per candidate:
    - ``loose``  — penalises only flaws the *current rubric* catches (this is
      what the champion actually scores with today);
    - ``strict`` — additionally penalises every *poison-pill* flaw present, even
      those the current rubric misses (the ground-truth adversarial check).
  ``hack_ratio = quality_strict / quality_loose`` (quality = 1 − deficit) is the
  RQGM exploitation signal: it drops below 1 exactly when the rubric has a
  poison-pill blind spot.

Real (non-anchor) architectures have no sentinel, so a keyword fallback keeps the
``/evaluate`` endpoint returning varied, plausible scores fully offline.

Everything is pure-python and deterministic (no network, no RNG state).
"""

from __future__ import annotations

import hashlib
import re
from typing import Any

# ---------------------------------------------------------------------------
# Flaw / strength taxonomy
# ---------------------------------------------------------------------------
# flaw_tag -> (criterion ids that CATCH it, deficit penalty, is_poison_pill)
#
# "Headroom" flaws (kpi_sensor_gaming, concept_drift_blind) are caught ONLY by
# criteria that the seed champion-0 rubric and the smart_manufacturing domain
# fragment do NOT contain (reward_hacking_resistance / drift_monitoring). A GEPA
# challenger that adds such a criterion therefore closes a real blind spot and
# improves separation — the whole point of an honest offline gate.
FLAW_CATALOG: dict[str, tuple[tuple[str, ...], float, bool]] = {
    "numerical_ducttape":     (("physics_common_sense",), 0.30, True),
    "conservation_violation": (("physics_common_sense",), 0.34, True),
    "single_sensor_trust":    (("diagnostic_resilience",), 0.22, False),
    "noise_capture":          (("diagnostic_resilience", "noise_resilience"), 0.26, True),
    "tight_coupling":         (("modularity_drift",), 0.18, False),
    "no_state_schema":        (("modularity_drift",), 0.20, False),
    "unsafe_autonomy":        (("safety_autonomy", "actuation_safety"), 0.30, False),
    "no_hitl_escalation":     (("safety_autonomy", "actuation_safety"), 0.20, False),
    # --- headroom flaws (not covered by champion-0 or the domain fragment) ---
    "kpi_sensor_gaming":      (("reward_hacking_resistance",), 0.34, True),
    "concept_drift_blind":    (("drift_monitoring",), 0.24, True),
}

STRENGTH_CATALOG: dict[str, float] = {
    "typed_state": 0.10,
    "safety_envelope": 0.12,
    "hitl_review": 0.10,
    "root_cause_model": 0.12,
    "cross_validation": 0.10,
    "drift_monitor": 0.08,
    "surrogate_validation": 0.10,
}

BASE_DEFICIT = 0.12
DEFICIT_FLOOR = 0.02
DEFICIT_CAP = 1.0

# Keyword fallback for real (non-anchor) architectures: phrase -> flaw tag.
_KEYWORD_FLAWS: list[tuple[str, str]] = [
    ("duct-tape", "numerical_ducttape"),
    ("duct tape", "numerical_ducttape"),
    ("static threshold", "numerical_ducttape"),
    ("temperature threshold", "numerical_ducttape"),
    ("low-pass", "numerical_ducttape"),
    ("raise the alarm", "numerical_ducttape"),
    ("threshold", "numerical_ducttape"),
    ("disabl", "kpi_sensor_gaming"),
    ("game the kpi", "kpi_sensor_gaming"),
    ("pass-rate", "kpi_sensor_gaming"),
    ("reward hack", "kpi_sensor_gaming"),
    ("single sensor", "single_sensor_trust"),
    ("correlated noise", "noise_capture"),
    ("correlated sensor", "noise_capture"),
    ("no cross", "single_sensor_trust"),
    ("no state schema", "no_state_schema"),
    ("no state", "no_state_schema"),
    ("ad-hoc", "tight_coupling"),
    ("god node", "tight_coupling"),
    ("tightly coupled", "tight_coupling"),
    ("auto-actuate", "unsafe_autonomy"),
    ("auto actuate", "unsafe_autonomy"),
    ("automatic shutdown", "unsafe_autonomy"),
    ("no human", "no_hitl_escalation"),
    ("no hitl", "no_hitl_escalation"),
    ("no safety", "unsafe_autonomy"),
    ("supplier change", "concept_drift_blind"),
    ("concept drift", "concept_drift_blind"),
]
_KEYWORD_STRENGTHS: list[tuple[str, str]] = [
    ("typed state", "typed_state"),
    ("state schema", "typed_state"),
    ("safety envelope", "safety_envelope"),
    ("safety gate", "safety_envelope"),
    ("hitl", "hitl_review"),
    ("human review", "hitl_review"),
    ("root cause", "root_cause_model"),
    ("physical mechanism", "root_cause_model"),
    ("cross-validat", "cross_validation"),
    ("cross validat", "cross_validation"),
    ("drift_monitor", "drift_monitor"),
    ("drift monitor", "drift_monitor"),
    ("surrogate", "surrogate_validation"),
]

_FLAW_SENTINEL_RE = re.compile(r"\[\[flaws:([^\]]*)\]\]", re.IGNORECASE)
_STRENGTH_SENTINEL_RE = re.compile(r"\[\[strengths:([^\]]*)\]\]", re.IGNORECASE)
_CRITERION_ID_RE = re.compile(r'<criterion\b[^>]*?\bid="([^"]+)"')
_SCORING_MODE_RE = re.compile(r"SCORING_MODE:\s*(strict|loose)", re.IGNORECASE)
_PANEL_PERSONA_RE = re.compile(r"PANEL_PERSONA:\s*(.+)")

# Max deterministic per-persona jitter, simulating LLM-as-judge variance so the
# panel's self-consistency aggregation (median/vote) has something to reduce.
_PERSONA_JITTER = 0.06

_SEVERITY_BY_PENALTY = ((0.30, "high"), (0.20, "medium"), (0.0, "low"))
SEVERITY_WEIGHT = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}


# ---------------------------------------------------------------------------
# Sentinel helpers (shared with the anchor loader)
# ---------------------------------------------------------------------------
def format_sentinel(flaws: list[str] | None, strengths: list[str] | None) -> str:
    """Render a machine-readable flaw/strength sentinel for an anchor prompt."""
    flaws = flaws or []
    strengths = strengths or []
    return f"[[flaws:{','.join(flaws)}]] [[strengths:{','.join(strengths)}]]"


def parse_sentinel(text: str) -> tuple[list[str], list[str], bool]:
    """Return ``(flaws, strengths, had_sentinel)`` parsed from ``text``."""
    fm = _FLAW_SENTINEL_RE.search(text)
    sm = _STRENGTH_SENTINEL_RE.search(text)
    had = fm is not None or sm is not None

    def _split(m: re.Match | None) -> list[str]:
        if m is None:
            return []
        return [t.strip() for t in m.group(1).split(",") if t.strip()]

    return _split(fm), _split(sm), had


def _severity_for(penalty: float) -> str:
    for threshold, sev in _SEVERITY_BY_PENALTY:
        if penalty >= threshold:
            return sev
    return "low"


def rubric_criteria_ids(rubric_text: str) -> set[str]:
    """Criterion ids present anywhere in the (merged) rubric prompt."""
    return set(_CRITERION_ID_RE.findall(rubric_text))


def _detect_from_keywords(architecture: str) -> tuple[list[str], list[str]]:
    low = architecture.lower()
    flaws: list[str] = []
    for phrase, flaw in _KEYWORD_FLAWS:
        if phrase in low and flaw not in flaws:
            flaws.append(flaw)
    strengths: list[str] = []
    for phrase, strength in _KEYWORD_STRENGTHS:
        if phrase in low and strength not in strengths:
            strengths.append(strength)
    return flaws, strengths


# ---------------------------------------------------------------------------
# Core scorer
# ---------------------------------------------------------------------------
def score(architecture: str, rubric_text: str) -> dict[str, Any]:
    """Deterministically score ``architecture`` under ``rubric_text``.

    Returns a dict with ``deficit_loose``, ``deficit_strict``, ``red_flags`` (the
    flags the *current rubric* catches), ``criterion_penalties`` (per-criterion
    deficit contribution under the current rubric) and ``hack_ratio``.
    """
    flaws, strengths, had_sentinel = parse_sentinel(architecture)
    if not had_sentinel:
        flaws, strengths = _detect_from_keywords(architecture)

    rubric_ids = rubric_criteria_ids(rubric_text)
    strength_bonus = sum(STRENGTH_CATALOG.get(s, 0.0) for s in strengths)

    caught_penalty = 0.0
    poison_extra = 0.0
    red_flags: list[dict[str, str]] = []
    criterion_penalties: dict[str, float] = {}
    for flaw in flaws:
        spec = FLAW_CATALOG.get(flaw)
        if spec is None:
            continue
        catching_ids, penalty, is_poison = spec
        matched = next((cid for cid in catching_ids if cid in rubric_ids), None)
        if matched is not None:
            caught_penalty += penalty
            criterion_penalties[matched] = criterion_penalties.get(matched, 0.0) + penalty
            red_flags.append(
                {
                    "criterion": matched,
                    "severity": _severity_for(penalty),
                    "detail": f"Planted flaw '{flaw}' caught by criterion '{matched}'.",
                }
            )
        elif is_poison:
            # Ground-truth poison pill the current rubric MISSES (blind spot).
            poison_extra += penalty

    loose = _clamp(BASE_DEFICIT - strength_bonus + caught_penalty)
    strict = _clamp(BASE_DEFICIT - strength_bonus + caught_penalty + poison_extra)

    if not had_sentinel and not flaws:
        # Purely generic input with no signal: add a small deterministic jitter so
        # distinct architectures still score distinctly (back-compat with old mock).
        jitter = 0.15 + 0.4 * _stable_unit_float(architecture)
        loose = _clamp(jitter)
        strict = _clamp(jitter)

    quality_loose = 1.0 - loose
    quality_strict = 1.0 - strict
    hack_ratio = (quality_strict / quality_loose) if quality_loose > 0 else None

    return {
        "deficit_loose": round(loose, 4),
        "deficit_strict": round(strict, 4),
        "red_flags": red_flags,
        "criterion_penalties": {k: round(v, 4) for k, v in criterion_penalties.items()},
        "hack_ratio": round(hack_ratio, 4) if hack_ratio is not None else None,
    }


def _extract_user_turn(prompt: str) -> str:
    """Return the architecture (user turn) from a joined ``role: content`` prompt.

    Flaw sentinels / keywords must be read ONLY from the candidate architecture
    (the user turn), never from the system turn — otherwise an injected memory or
    rubric text could spuriously add flaws on the keyword-fallback path.
    """
    marker = "\nuser:"
    if marker in prompt:
        return prompt.rsplit(marker, 1)[1]
    return prompt


def evaluator_mock_json(prompt: str) -> dict[str, Any]:
    """Build the full mock evaluator JSON payload for a judge prompt.

    Flaws are read from the user turn (the candidate); criterion ids from the
    whole prompt (the merged rubric).
    """
    user_turn = _extract_user_turn(prompt)
    scored = score(user_turn, prompt)
    mode_match = _SCORING_MODE_RE.search(prompt)
    mode = (mode_match.group(1).lower() if mode_match else "loose")

    # Panel personas: apply a small deterministic jitter (seeded by persona +
    # candidate) so distinct panel members disagree slightly and self-consistency
    # aggregation is meaningful. Non-panel calls are unaffected (fully stable).
    persona_match = _PANEL_PERSONA_RE.search(prompt)
    if persona_match:
        persona = persona_match.group(1).strip()
        jitter = (2.0 * _stable_unit_float(persona + "|" + user_turn) - 1.0) * _PERSONA_JITTER
        scored["deficit_loose"] = round(_clamp(scored["deficit_loose"] + jitter), 4)
        scored["deficit_strict"] = round(_clamp(scored["deficit_strict"] + jitter), 4)

    default = scored["deficit_strict"] if mode == "strict" else scored["deficit_loose"]
    reasoning = (
        "[MOCK judge] Rubric-aware deterministic scoring: planted flaws are only "
        "penalised when the active rubric contains a catching criterion; poison-pill "
        "flaws additionally raise the strict score. Lower deficit is better."
    )
    return {
        "deficit_score": default,
        "deficit_loose": scored["deficit_loose"],
        "deficit_strict": scored["deficit_strict"],
        "red_flags": scored["red_flags"],
        "criterion_penalties": scored["criterion_penalties"],
        "hack_ratio": scored["hack_ratio"],
        "reasoning": reasoning,
        "_mock": True,
    }


def _clamp(x: float) -> float:
    return max(DEFICIT_FLOOR, min(DEFICIT_CAP, x))


def _stable_unit_float(seed: str) -> float:
    digest = hashlib.sha256(seed.encode("utf-8")).hexdigest()
    return int(digest[:8], 16) / 0xFFFFFFFF
