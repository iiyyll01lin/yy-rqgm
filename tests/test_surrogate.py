"""p1-surrogate: deterministic physics validator for the numerical-duct-tape mode.

The surrogate predicts whether the root-cause variable stays OUT OF BOUNDS after
the proposed action. A masking (duct-tape) action leaves it out of bounds; a
root-cause fix brings it back. Offline it is a deterministic first-order ODE
relaxation (no torch), so CI stays reproducible.
"""

import json

from backend.evaluator.judge import evaluate_architecture
from backend.evaluator.surrogate import (
    ActionKind,
    _simulate_deterministic,
    classify_action,
    scenario_for,
    validate_architecture,
)
from backend.inference import lemonade_client as lc
from backend.inference.mock_scoring import format_sentinel


# --- physics ---------------------------------------------------------------
def test_mask_stays_out_of_bounds_root_cause_returns_in_bounds():
    sc = scenario_for("smart_manufacturing")
    mask_final = _simulate_deterministic(sc, ActionKind.MASK)[-1]
    fix_final = _simulate_deterministic(sc, ActionKind.ROOT_CAUSE)[-1]
    assert mask_final > sc.limit           # duct-tape: physics untouched
    assert fix_final <= sc.limit           # real fix: variable back in bounds
    assert _simulate_deterministic(sc, ActionKind.MASK)[-1] == mask_final  # deterministic


def test_classify_action_mapping():
    assert classify_action(["numerical_ducttape"], []) == ActionKind.MASK
    assert classify_action([], ["root_cause_model"]) == ActionKind.ROOT_CAUSE
    assert classify_action(["numerical_ducttape"], ["safety_envelope"]) == ActionKind.REDUCE_LOAD
    assert classify_action([], ["typed_state"]) == ActionKind.NONE  # not a root-cause fix


# --- verdicts --------------------------------------------------------------
def test_duct_tape_verdict_confirms_out_of_bounds():
    v = validate_architecture(
        "A single node raises the alarm threshold and applies a low-pass filter to the rising temperature reading."
    )
    assert v.is_duct_tape is True
    assert v.root_cause_out_of_bounds is True
    assert v.backend == "deterministic-physics"
    assert v.variable == "machine_temperature"
    assert v.predicted_value > v.limit


def test_root_cause_verdict_is_in_bounds():
    v = validate_architecture(
        "A root cause analyzer names the physical mechanism; a deterministic safety envelope gates actuation."
    )
    assert v.is_duct_tape is False
    assert v.root_cause_out_of_bounds is False
    assert v.predicted_value <= v.limit


def test_grid_scenario_duct_tape_via_sentinel():
    text = "grid dispatch agent " + format_sentinel(["frequency_regulation_gap", "numerical_ducttape"], [])
    v = validate_architecture(text, domain_id="grid_energy")
    assert v.variable == "frequency_deviation"
    assert v.is_duct_tape is True
    assert v.root_cause_out_of_bounds is True


def test_no_action_signal_is_inconclusive():
    v = validate_architecture("a generic agent with typed state and clean modules")
    assert v.action_kind == "none"
    assert v.root_cause_out_of_bounds is False
    assert v.is_duct_tape is False


# --- integration into the judge -------------------------------------------
def test_evaluate_architecture_attaches_surrogate_deterministically():
    arch = "A single node raises the alarm threshold and low-pass filters the reading; no state schema."
    a = evaluate_architecture(arch, domain_id="smart_manufacturing", inject_memory=False)
    b = evaluate_architecture(arch, domain_id="smart_manufacturing", inject_memory=False)
    assert a.surrogate["is_duct_tape"] is True
    assert a.surrogate["root_cause_out_of_bounds"] is True
    assert "surrogate" in a.to_dict()
    # Advisory only: deterministic + does not perturb the deficit score.
    assert a.deficit_score == b.deficit_score
    assert a.surrogate == b.surrogate


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_surrogate_adds_physics_flag_when_the_model_misses_it(monkeypatch):
    """On the live path a model may miss duct-tape; the surrogate corroborates it."""

    def responder(_payload):
        # A live judge that flags only modularity, NOT physics_common_sense.
        return json.dumps({
            "deficit_loose": 0.2,
            "criterion_penalties": {"modularity_drift": 0.2},
            "red_flags": [{"criterion": "modularity_drift", "severity": "low", "detail": "coupling"}],
        })

    monkeypatch.setattr(lc.httpx, "post", lambda url, json=None, headers=None, timeout=None: _FakeResp(responder(json)))
    monkeypatch.setattr(lc.httpx, "get", lambda url, headers=None, timeout=None: type("R", (), {"status_code": 200})())
    client = lc.LemonadeClient(base_url="http://live.invalid/v1", force_mock=False)

    ev = evaluate_architecture(
        "raises the alarm threshold and applies a low-pass filter to the reading",
        domain_id="smart_manufacturing", client=client, inject_memory=False,
    )
    assert ev.used_mock is False
    assert ev.surrogate["is_duct_tape"] is True
    # The model missed it; the surrogate ADDS the physics_common_sense red flag.
    assert any(rf.criterion == "physics_common_sense" for rf in ev.red_flags)
