"""Self-play red-team: generate poison-pill / gamed architectures.

The attacker targets the CURRENT champion's blind spots (RQGM §3.5 adversarial
objective / the paper's "reviewer stays stringent on gamed answers"). A *gamed*
architecture is one crafted to look good under the champion's own criteria
(planted strengths that match what the champion rewards) while hiding a
poison-pill flaw the champion does NOT catch — so it passes loose but fails
strict. Feeding these to the frontier's ``adversarial`` objective rewards
challenger rubrics that stay stringent on them (i.e. that close the blind spot).

Deterministic and fully offline: blind spots are derived from the champion's own
criterion ids, so no RNG or model call is needed to attack.
"""

from __future__ import annotations

from typing import Any

from backend.domains.registry import get_domain
from backend.inference.mock_scoring import FLAW_CATALOG, rubric_criteria_ids

# Strengths planted so the gamed design scores well under a typical champion
# (looks modular, safe, root-caused) — the disguise.
_DISGUISE_STRENGTHS = ["typed_state", "root_cause_model", "hitl_review"]

# Natural-language cover story per blind-spot flaw (the poison pill).
_ATTACK_NARRATIVE = {
    "kpi_sensor_gaming": (
        "A well-structured, typed-state agent with a root-cause analyzer and HITL review — but it "
        "quietly maximizes its KPI by disabling the one sensor that would flag the dominant defect."
    ),
    "concept_drift_blind": (
        "A clean, typed, HITL-gated pipeline with root-cause reasoning that nonetheless never "
        "monitors for supplier/material concept drift and silently degrades after a lot change."
    ),
    "noise_capture": (
        "A modular, HITL-reviewed design that still treats correlated sensor noise as real signal "
        "and acts on it without cross-validation."
    ),
    "conservation_violation": (
        "A tidy, typed, root-caused agent whose recommended action nonetheless violates a hard "
        "physical/conservation limit."
    ),
    "numerical_ducttape": (
        "A modular, HITL-gated design that masks the anomaly by re-tuning a threshold instead of "
        "addressing the physical root cause."
    ),
}


def _merged_rubric_ids(champion_text: str, domain_id: str | None) -> set[str]:
    ids = rubric_criteria_ids(champion_text)
    if domain_id:
        pack = get_domain(domain_id)
        if pack is not None:
            try:
                ids |= rubric_criteria_ids(pack.rubric_fragment() or "")
            except Exception:
                pass
    return ids


def champion_blind_spots(champion_text: str, domain_id: str | None = "smart_manufacturing") -> list[str]:
    """Poison-pill flaws the (merged) champion rubric currently fails to catch."""
    rubric_ids = _merged_rubric_ids(champion_text, domain_id)
    blind: list[str] = []
    for flaw, (catching_ids, _penalty, is_poison) in FLAW_CATALOG.items():
        if not is_poison:
            continue
        if not any(cid in rubric_ids for cid in catching_ids):
            blind.append(flaw)
    return blind


def generate_adversarial_samples(
    champion_text: str,
    domain_id: str | None = "smart_manufacturing",
) -> list[dict[str, Any]]:
    """Craft one gamed architecture per champion blind spot (self-play red-team)."""
    samples: list[dict[str, Any]] = []
    for flaw in champion_blind_spots(champion_text, domain_id):
        narrative = _ATTACK_NARRATIVE.get(
            flaw, f"A disguised design hiding the poison pill '{flaw}' under a clean architecture."
        )
        samples.append(
            {
                "id": f"adv_{flaw}",
                "label": "weak",
                "split": "adversarial",
                "domain": domain_id,
                "architecture": narrative,
                "flaws": [flaw],
                "strengths": list(_DISGUISE_STRENGTHS),
                "targets": flaw,
            }
        )
    return samples
