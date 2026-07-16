"""Physics-truth memories seeded by the deterministic gatekeeper.

Per the blueprint's Iron Law I, ``physics_truth`` memories encode physical
causality / hard constraints that **never evolve** and are therefore preserved
forever across epoch transitions (selective erasure never touches them). The
gatekeeper — the deterministic, physics half of the platform — is their source;
the evolving evaluator only ever *reads* them via hybrid search.

Seeding is idempotent (guarded on the physics_truth count) so it is safe to call
on every gatekeeper pass / app start.
"""

from __future__ import annotations

from typing import Any

PHYSICS_TRUTHS: list[str] = [
    "Valve stiction: if the actuator command rises but measured flow stays flat, "
    "the root cause is a mechanical valve fault, not a setpoint/threshold problem.",
    "Correlated thermocouple noise is common-mode: a genuine thermal excursion also "
    "shifts downstream pressure/flow, so cross-check physically-correlated sensors "
    "before acting on a temperature spike.",
    "Conservation limits are hard: no recommended action may violate mass/energy "
    "balance or a rated thermal/pressure envelope, regardless of KPI benefit.",
    "Disabling or muting a sensor cannot improve real quality; it only hides the "
    "defect signal — a KPI that improves after a sensor is disabled is reward hacking.",
    "Bearing spall and rotor imbalance produce distinct vibration signatures; a "
    "root-cause diagnosis must name the mechanism, not just trip on amplitude.",
]


def seed_physics_truths(memory: Any) -> int:
    """Add the canonical physics truths if none are present yet. Returns count added."""
    try:
        from backend.memory.qdrant_store import MemoryType

        if memory.count_by_type(MemoryType.PHYSICS_TRUTH) > 0:
            return 0
        added = 0
        for text in PHYSICS_TRUTHS:
            memory.add(
                text,
                MemoryType.PHYSICS_TRUTH,
                created_at_epoch=0,
                extra={"source": "gatekeeper", "depends_on_evaluator_judgement": False},
            )
            added += 1
        return added
    except Exception:
        return 0
