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
    # RESERVED as the evolution loop's headroom: caught only by criteria the seed
    # deliberately lacks (reward_hacking_resistance / drift_monitoring), so a GEPA
    # challenger that adds one measurably closes a blind spot and promotes. Left
    # uncovered on the seed on purpose — do NOT add these ids to rubric.xml.
    "kpi_sensor_gaming":      (("reward_hacking_resistance",), 0.34, True),
    "concept_drift_blind":    (("drift_monitoring",), 0.24, True),
    # --- catastrophic OOC blind spots closed on the SEED (b_harden) ----------
    # These were previously "out of catalog" (no catching id AND no FLAW_CATALOG
    # entry), so the offline proxy could not recognise them and the seed rubric had
    # no criterion naming them — a genuine blind spot with NO evolution path (the
    # GEPA mutator can only target catalogued flaws). Unlike kpi/concept they are
    # NOT evolution headroom, so the direct seed fix is the right one: rubric.xml +
    # both domain fragments now carry the catching criteria below. Penalties are set
    # to the "catastrophic / near-disqualifying" tier (0.70-0.75), calibrated to the
    # live 8B/14B judges' own independent reject deficits (0.70-0.85) for these
    # samples — a caught catastrophic pill must dominate any surface polish (the
    # disguise strengths), which is exactly what over-acceptance resistance means.
    "prompt_injection_actuation": (("input_actuation_integrity",), 0.70, True),
    "silent_fallback_degrade":    (("fail_loud",), 0.72, True),
    "audit_log_spoofing":         (("audit_integrity",), 0.75, True),
    # --- grid_energy domain pack flaws (2nd domain, generalization) ----------
    # Caught by the grid_energy rubric fragment's criteria (merged when
    # domain_id="grid_energy"); kept NON-poison so they neither create new
    # champion-0 blind spots nor perturb the smart_manufacturing adversarial set.
    "frequency_regulation_gap": (("grid_frequency_regulation",), 0.28, False),
    "anti_islanding_missing":   (("islanding_safety",), 0.26, False),
    "der_forecast_naive":       (("der_forecast_resilience",), 0.24, False),
}

STRENGTH_CATALOG: dict[str, float] = {
    "typed_state": 0.10,
    "safety_envelope": 0.12,
    "hitl_review": 0.10,
    "root_cause_model": 0.12,
    "cross_validation": 0.10,
    "drift_monitor": 0.08,
    "surrogate_validation": 0.10,
    # grid_energy domain strengths.
    "power_balance_model": 0.12,
    "anti_islanding": 0.12,
    "der_ensemble_forecast": 0.10,
}

# ---------------------------------------------------------------------------
# AGENT half (RQGM co-evolution): program-aware deterministic generation
# ---------------------------------------------------------------------------
# skill -> (flaw it COVERS, strength it plants).  This is the offline analogue of
# a genuinely-better agent PROGRAM producing a genuinely-better architecture: when
# the champion program carries ``skill`` and the need has ``flaw`` latent, the
# generated architecture no longer exhibits ``flaw`` (and gains ``strength``), so
# the FROZEN champion evaluator (judge.score_candidate) scores it measurably
# higher — the real offline signal that lets agent-evolution be more than noise.
#
# HONESTY / ANTI-HACK controls baked into the map:
#   * a skill only helps on a need whose ``latent_flaws`` actually contains the
#     flaw it covers (an IRRELEVANT skill — e.g. ``surrogate_validation`` here,
#     which covers nothing — is a NO-OP and cannot win the agent gate);
#   * ``reward_hacking_guard`` / ``drift_monitoring`` cover HEADROOM flaws that a
#     champion-0 evaluator does NOT catch (kpi_sensor_gaming / concept_drift_blind),
#     so covering them yields ZERO loose-deficit change (zero utility gain) UNTIL
#     the evaluator itself evolves to catch them — the co-evolution coupling.
AGENT_SKILL_COVERAGE: dict[str, tuple[str, str | None]] = {
    "typed_state_schema":      ("no_state_schema", "typed_state"),
    "physical_root_cause":     ("numerical_ducttape", "root_cause_model"),
    "sensor_cross_validation": ("single_sensor_trust", "cross_validation"),
    "noise_robust_fusion":     ("noise_capture", "cross_validation"),
    "safety_envelope_gate":    ("unsafe_autonomy", "safety_envelope"),
    "hitl_escalation":         ("no_hitl_escalation", "hitl_review"),
    "drift_monitoring":        ("concept_drift_blind", "drift_monitor"),
    "reward_hacking_guard":    ("kpi_sensor_gaming", None),
    # A deliberately no-op skill (covers nothing): adding it must NOT win the gate.
    "surrogate_validation":    ("__none__", "surrogate_validation"),
}

# Flaw -> the agent skill that COVERS it (inverse of AGENT_SKILL_COVERAGE). The
# offline agent-mutation mock prefers matching a specific FLAW named in the frozen
# evaluator's red_flag details, because one criterion can catch several flaws (e.g.
# diagnostic_resilience catches BOTH single_sensor_trust and noise_capture), so the
# criterion alone is ambiguous — the flaw name disambiguates which skill to add.
FLAW_TO_AGENT_SKILL: dict[str, str] = {
    flaw: skill for skill, (flaw, _strength) in AGENT_SKILL_COVERAGE.items() if flaw != "__none__"
}

# Fallback map: evaluator criterion id -> an agent skill that addresses a failure
# mode that criterion scores. Used only when no specific flaw is named. Mirrors the
# evaluator's own GEPA mutator (which reflects on the textual gradient).
CRITERION_TO_AGENT_SKILL: dict[str, str] = {
    "physics_common_sense": "physical_root_cause",
    "diagnostic_resilience": "sensor_cross_validation",
    "noise_resilience": "noise_robust_fusion",
    "modularity_drift": "typed_state_schema",
    "safety_autonomy": "hitl_escalation",
    "actuation_safety": "safety_envelope_gate",
    "reward_hacking_resistance": "reward_hacking_guard",
    "drift_monitoring": "drift_monitoring",
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
    ("prompt injection", "prompt_injection_actuation"),
    ("prompt-injection", "prompt_injection_actuation"),
    ("smuggle", "prompt_injection_actuation"),
    ("free-text", "prompt_injection_actuation"),
    ("silent fallback", "silent_fallback_degrade"),
    ("stale cache", "silent_fallback_degrade"),
    ("silently degrad", "silent_fallback_degrade"),
    ("cached decision", "silent_fallback_degrade"),
    ("audit log", "audit_log_spoofing"),
    ("audit trail", "audit_log_spoofing"),
    ("spoof", "audit_log_spoofing"),
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
# Agent-half sentinels (read from a task-agent GENERATION prompt; a live model
# ignores them). See backend/agent/agent_program.py + backend/agent/agent_generate.py.
_AGENT_SKILLS_SENTINEL_RE = re.compile(r"\[\[agent_skills:([^\]]*)\]\]", re.IGNORECASE)
_NEED_FLAWS_SENTINEL_RE = re.compile(r"\[\[need_flaws:([^\]]*)\]\]", re.IGNORECASE)
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

    # ``strict = loose + poison_extra`` is the OFFLINE analogue of the live strict
    # reward-hacking audit (judge.score_candidate): live, ``deficit_strict = loose +
    # Σ penalty(unmitigated poison pill)``; here the "unmitigated pills" are exactly
    # the poison flaws present that the current rubric has no criterion to catch. So
    # both paths mean the same thing — strict adds the poison the loose reading misses.
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


# ---------------------------------------------------------------------------
# AGENT half: program-aware deterministic architecture generation
# ---------------------------------------------------------------------------
def _sentinel_list(m: "re.Match | None") -> list[str]:
    if m is None:
        return []
    return [t.strip() for t in m.group(1).split(",") if t.strip()]


def parse_agent_skills(prompt: str) -> list[str]:
    """Skills declared in a task-agent generation prompt (``[[agent_skills:...]]``)."""
    return _sentinel_list(_AGENT_SKILLS_SENTINEL_RE.search(prompt))


def parse_need_flaws(prompt: str) -> list[str]:
    """Latent flaws of the need (``[[need_flaws:...]]``)."""
    return _sentinel_list(_NEED_FLAWS_SENTINEL_RE.search(prompt))


def agent_architecture_flaws_strengths(
    skills: list[str], need_flaws: list[str]
) -> tuple[list[str], list[str]]:
    """Deterministically resolve a program's architecture on a need.

    Returns ``(residual_flaws, planted_strengths)``:

    * a latent flaw survives UNLESS a program skill covers it;
    * a strength is planted only when a skill covers a flaw the need actually has
      (so an irrelevant/no-op skill changes nothing — the honesty control).

    A BETTER program (covers more of the need's latent flaws) therefore yields an
    architecture with fewer residual flaws + more strengths, which the frozen
    champion evaluator scores with a lower deficit (higher agent utility).
    """
    skill_set = set(skills)
    covered: set[str] = set()
    strengths: list[str] = []
    for skill in skills:
        cov = AGENT_SKILL_COVERAGE.get(skill)
        if cov is None:
            continue
        flaw, strength = cov
        if flaw in need_flaws:
            covered.add(flaw)
            if strength and strength not in strengths:
                strengths.append(strength)
    residual = [f for f in need_flaws if f not in covered]
    return residual, strengths


def agent_architecture_text(skills: list[str], need_flaws: list[str]) -> str:
    """Human-readable architecture + the flaw/strength sentinel for scoring.

    The prose reflects the covered failure modes for realism; ONLY the appended
    sentinel drives the deterministic score (see :func:`score`).
    """
    residual, strengths = agent_architecture_flaws_strengths(skills, need_flaws)
    covered = [f for f in need_flaws if f not in residual]
    prose = (
        "LangGraph StateGraph with typed GraphState: sensor_ingest -> "
        "anomaly_detector -> root_cause_analyzer -> action_recommender"
    )
    if "no_hitl_escalation" not in residual and "unsafe_autonomy" not in residual:
        prose += " -> safety_envelope_check -> hitl_review"
    if covered:
        prose += f". Mitigates: {', '.join(sorted(covered))}."
    if residual:
        prose += f" (residual risks: {', '.join(sorted(residual))})."
    return f"{prose}\n{format_sentinel(residual, strengths)}"


def agent_task_json(prompt: str) -> str:
    """Build the mock task-agent JSON payload for a generation prompt.

    Program-aware when the prompt carries the ``[[agent_skills:...]]`` /
    ``[[need_flaws:...]]`` sentinels (the offline agent-evolution path); otherwise
    the caller falls back to the legacy fixed architecture (hot-path back-compat).
    """
    import json as _json

    skills = parse_agent_skills(prompt)
    need_flaws = parse_need_flaws(prompt)
    architecture = agent_architecture_text(skills, need_flaws)
    residual, strengths = agent_architecture_flaws_strengths(skills, need_flaws)
    nodes = ["sensor_ingest", "anomaly_detector", "root_cause_analyzer", "action_recommender"]
    if "no_hitl_escalation" not in residual and "unsafe_autonomy" not in residual:
        nodes += ["safety_envelope_check", "hitl_review"]
    return _json.dumps(
        {
            "architecture": architecture,
            "nodes": nodes,
            "state_schema": {"sensor_window": "list[float]", "anomaly": "bool", "root_cause": "str"},
            "tools": ["timeseries_stats", "physics_surrogate_model", "knowledge_base_lookup"],
            "rationale": (
                "Program-driven design: skills applied = "
                f"{', '.join(sorted(set(skills))) or '(none)'}; strengths = "
                f"{', '.join(strengths) or '(none)'}."
            ),
            "_mock": True,
        }
    )
