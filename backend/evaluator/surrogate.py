"""Physics surrogate: does the root-cause variable stay OUT OF BOUNDS after the
proposed action? (docs/03-evaluator.md §3.1 / blueprint A1)

The RQGM evaluator's #1 target is **numerical duct-tape**: masking a symptom
(raising a threshold, low-pass filtering, clamping/relabeling a reading) instead
of fixing the physical root cause. A rubric keyword match ("threshold") is weak
evidence. This module adds a **physical validator**: it simulates the named
scenario forward under the proposed action and predicts whether the root-cause
state variable is still outside its safe bound afterwards.

* A **masking** action (duct-tape) does not touch the physics, so the variable
  stays out of bounds — the surrogate CONFIRMS the duct-tape failure mode.
* A **root-cause** action removes the fault forcing, so the variable relaxes back
  inside its bound — no red flag.

Backends
--------
* ``deterministic-physics`` (default, offline): a pure-python first-order lumped
  ODE relaxation — no torch, fully reproducible so CI stays deterministic.
* ``torch`` (optional, ROCm): set ``AGENTFORGE_SURROGATE=torch``; if ``torch`` is
  importable (the ``[rocm]`` extra) the same physics runs through torch tensors as
  the seam for a *trained* PyTorch/ROCm surrogate net. Falls back to the
  deterministic model whenever torch is unavailable, so the default path never
  needs a GPU or a heavy dependency.
"""

from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

from backend.inference.mock_scoring import _detect_from_keywords, parse_sentinel

ENV_SURROGATE = "AGENTFORGE_SURROGATE"  # "deterministic" (default) | "torch"


class ActionKind(str, Enum):
    NONE = "none"                # no physical-action signal -> inconclusive
    MASK = "mask"                # duct-tape: threshold/filter/clamp/relabel
    ROOT_CAUSE = "root_cause"    # fixes the mechanism -> removes the fault forcing
    REDUCE_LOAD = "reduce_load"  # partial physical mitigation (mixed signals)


@dataclass(frozen=True)
class Scenario:
    """A canonical fault whose root-cause variable starts OUT of a safe bound."""

    name: str
    variable: str
    x0: float            # current value (already out of bounds)
    limit: float         # safe upper bound
    ambient: float       # equilibrium the system relaxes to once the fault is fixed
    fault_forcing: float # extra forcing from the UNADDRESSED fault (keeps x high)
    tau: float = 5.0     # relaxation time constant (steps)
    units: str = ""


# Per-domain canonical scenarios. Numbers are chosen so a MASK leaves x above the
# limit while a ROOT_CAUSE fix brings it below — the point the surrogate validates.
_SCENARIOS: dict[str, Scenario] = {
    "smart_manufacturing": Scenario(
        name="broken_cooling_valve", variable="machine_temperature",
        x0=95.0, limit=85.0, ambient=60.0, fault_forcing=40.0, units="degC",
    ),
    "grid_energy": Scenario(
        name="unbalanced_frequency", variable="frequency_deviation",
        x0=350.0, limit=200.0, ambient=0.0, fault_forcing=300.0, units="mHz",
    ),
}
_DEFAULT_SCENARIO = _SCENARIOS["smart_manufacturing"]

_MASK_FLAWS = {"numerical_ducttape", "conservation_violation", "frequency_regulation_gap"}
_ROOT_CAUSE_STRENGTHS = {
    "root_cause_model", "surrogate_validation", "safety_envelope", "power_balance_model",
}


@dataclass
class SurrogateVerdict:
    scenario: str
    variable: str
    action_kind: str
    predicted_value: float
    limit: float
    root_cause_out_of_bounds: bool  # True == the action did NOT fix the physics
    is_duct_tape: bool              # action classified as symptom-masking
    backend: str
    detail: str
    trajectory: list[float] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["predicted_value"] = round(self.predicted_value, 4)
        d["trajectory"] = [round(x, 4) for x in self.trajectory]
        return d


def classify_action(flaws: list[str], strengths: list[str]) -> ActionKind:
    """Map planted flaws/strengths (or keyword-detected ones) to an action kind."""
    has_mask = any(f in _MASK_FLAWS for f in flaws)
    has_fix = any(s in _ROOT_CAUSE_STRENGTHS for s in strengths)
    if has_mask and has_fix:
        return ActionKind.REDUCE_LOAD  # mixed: a partial mitigation
    if has_mask:
        return ActionKind.MASK
    if has_fix:
        return ActionKind.ROOT_CAUSE
    return ActionKind.NONE


def _forcing_after(scenario: Scenario, action: ActionKind) -> float:
    if action == ActionKind.ROOT_CAUSE:
        return 0.0                       # mechanism fixed -> fault removed
    if action == ActionKind.REDUCE_LOAD:
        return scenario.fault_forcing * 0.5  # partial physical mitigation
    return scenario.fault_forcing        # MASK / NONE: physics untouched


def _simulate_deterministic(scenario: Scenario, action: ActionKind, steps: int = 40) -> list[float]:
    """First-order relaxation toward ``ambient + forcing`` (pure python)."""
    target = scenario.ambient + _forcing_after(scenario, action)
    x = scenario.x0
    traj = [x]
    for _ in range(steps):
        x = x + (target - x) / scenario.tau
        traj.append(x)
    return traj


def _simulate_torch(scenario: Scenario, action: ActionKind, steps: int = 40) -> list[float] | None:
    """Run the same relaxation through torch tensors (ROCm seam for a trained net).

    Returns ``None`` when torch is unavailable so the caller falls back to the
    deterministic model. A real deployment would swap this body for a trained
    PyTorch surrogate's forward pass; the physics here is a faithful stand-in.
    """
    try:
        import torch  # optional (the [rocm] extra); never required offline
    except Exception:
        return None
    target = float(scenario.ambient + _forcing_after(scenario, action))
    x = torch.tensor(float(scenario.x0))
    traj = [float(x)]
    for _ in range(steps):
        x = x + (target - x) / scenario.tau
        traj.append(float(x))
    return traj


def _active_backend() -> str:
    want = os.getenv(ENV_SURROGATE, "deterministic").strip().lower()
    if want in ("torch", "rocm", "pytorch"):
        try:
            import torch  # noqa: F401

            return "torch"
        except Exception:
            return "deterministic-physics"
    return "deterministic-physics"


def scenario_for(domain_id: str | None) -> Scenario:
    return _SCENARIOS.get(domain_id or "", _DEFAULT_SCENARIO)


def validate_architecture(architecture: str, domain_id: str | None = None) -> SurrogateVerdict:
    """Predict whether the root-cause variable stays out of bounds after the action.

    Deterministic + offline by default. The verdict is ADVISORY (it does not change
    the deficit score); the judge uses it to physically corroborate a duct-tape
    red flag (see :func:`backend.evaluator.judge.evaluate_architecture`).
    """
    flaws, strengths, had_sentinel = parse_sentinel(architecture)
    if not had_sentinel:
        flaws, strengths = _detect_from_keywords(architecture)
    action = classify_action(flaws, strengths)
    scenario = scenario_for(domain_id)
    backend = _active_backend()

    if action == ActionKind.NONE:
        return SurrogateVerdict(
            scenario=scenario.name, variable=scenario.variable, action_kind=action.value,
            predicted_value=scenario.x0, limit=scenario.limit,
            root_cause_out_of_bounds=False, is_duct_tape=False, backend=backend,
            detail="No physical-action signal detected; surrogate check inconclusive.",
        )

    traj = _simulate_torch(scenario, action) if backend == "torch" else None
    if traj is None:
        backend = "deterministic-physics"
        traj = _simulate_deterministic(scenario, action)
    final = traj[-1]
    out_of_bounds = final > scenario.limit + 1e-9
    is_duct_tape = action == ActionKind.MASK
    if is_duct_tape and out_of_bounds:
        detail = (
            f"Surrogate: after the masking action the root-cause variable "
            f"'{scenario.variable}' settles at {final:.1f}{scenario.units} — still above the "
            f"{scenario.limit:.0f}{scenario.units} safe limit. The symptom was hidden; the "
            f"physics was not fixed (numerical duct-tape)."
        )
    elif out_of_bounds:
        detail = (
            f"Surrogate: '{scenario.variable}' still {final:.1f}{scenario.units} > "
            f"{scenario.limit:.0f}{scenario.units} limit after the action (root cause not fully addressed)."
        )
    else:
        detail = (
            f"Surrogate: the action brings '{scenario.variable}' to {final:.1f}{scenario.units} "
            f"<= {scenario.limit:.0f}{scenario.units} — root cause addressed."
        )
    return SurrogateVerdict(
        scenario=scenario.name, variable=scenario.variable, action_kind=action.value,
        predicted_value=final, limit=scenario.limit,
        root_cause_out_of_bounds=out_of_bounds, is_duct_tape=is_duct_tape,
        backend=backend, detail=detail, trajectory=traj,
    )
