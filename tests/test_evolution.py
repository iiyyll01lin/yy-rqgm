"""Tests for the RQGM/GEPA evolution loop: two-stage code gate, hack-ratio
tolerance tightening, selective erasure (soft-delete + reconfirm), versioning."""

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import evolve, gate, rqgm_adapter, versioning
from backend.evaluator.mutation import mutate_rubric_text
from backend.memory.qdrant_store import EvolutionaryMemory, MemoryType


def test_propose_challenger_registers_and_scores():
    prop = evolve.propose_challenger()
    assert prop.version.startswith("challenger-")
    assert prop.new_criteria  # always yields a concrete change
    assert "champion_separation" in prop.metrics
    assert "challenger_separation" in prop.metrics
    assert prop.rubric_diff  # a non-empty unified diff
    assert versioning.get_challenger(prop.version) is not None


# ---------------------------------------------------------------------------
# CODE GATE — the anti-reward-hacking core (runs BEFORE HITL)
# ---------------------------------------------------------------------------
def test_worse_challenger_rejected_by_code_gate():
    """A strictly-worse challenger must be rejected by the code gate even when
    the human approves — HITL can never override a failed gate."""
    champion = versioning.get_champion_rubric_text()
    # A challenger that catches FEWER flaws: drop the safety_autonomy criterion so
    # it detects strictly less on weak anchors -> lower separation.
    worse = champion.replace(
        '<criterion id="safety_autonomy" weight="0.15">', '<criterion id="_removed_safety" weight="0.15">'
    )
    versioning.register_challenger(version="challenger-worse", rubric_text=worse, parent_version="champion-0")

    result = evolve.approve_challenger("challenger-worse", approve=True)
    assert result["applied"] is False
    assert result["gate"]["passed"] is False
    assert result["hitl"]["consulted"] is False  # HITL never even asked
    assert versioning.get_epoch() == 0


def test_tie_favors_incumbent():
    """An identical challenger (tie) must NOT be promoted (tie favours incumbent)."""
    champion = versioning.get_champion_rubric_text()
    versioning.register_challenger(version="challenger-tie", rubric_text=champion, parent_version="champion-0")
    result = evolve.approve_challenger("challenger-tie", approve=True)
    assert result["applied"] is False
    assert result["gate"]["p1_non_inferior"] is False
    assert result["gate"]["separation_delta"] == 0.0
    assert versioning.get_epoch() == 0


def test_better_challenger_passes_gate_and_hitl_promotes():
    prop = evolve.propose_challenger()  # adds reward_hacking_resistance (closes a blind spot)
    result = evolve.approve_challenger(prop.version, approve=True, memory=EvolutionaryMemory(collection="t_gate"))
    assert result["gate"]["passed"] is True
    assert result["applied"] is True
    assert result["epoch_id"] == 1
    assert versioning.get_epoch() == 1


def test_gate_pass_but_hitl_veto_does_not_promote():
    """HITL is a final safety veto: it may reject a challenger that PASSED the gate."""
    prop = evolve.propose_challenger()
    result = evolve.approve_challenger(prop.version, approve=False)
    assert result["gate"]["passed"] is True
    assert result["applied"] is False
    assert result["hitl"] == {"consulted": True, "approved": False, "vetoed": True}
    assert versioning.get_epoch() == 0


# ---------------------------------------------------------------------------
# RQGM hack-ratio -> tolerance tightening
# ---------------------------------------------------------------------------
def test_hack_ratio_detection_tightens():
    """Low strict/loose quality ratio (poison-pill blind spot) tightens tolerances."""
    # Gamed samples: high loose quality, low strict quality -> hack ratio well below 0.6.
    strict_q = [0.2, 0.15, 0.25, 0.1]
    loose_q = [0.9, 0.95, 0.9, 0.92]
    rep = rqgm_adapter.detect_exploitation(strict_q, loose_q)
    assert rep.exploitation_detected is True
    assert rep.tightened is True
    assert len(rep.tolerances_after) < len(rep.tolerances_before)

    # Honest samples: strict ~ loose -> ratio ~1 -> no tightening.
    ok = rqgm_adapter.detect_exploitation([0.85, 0.9, 0.88], [0.9, 0.92, 0.9])
    assert ok.exploitation_detected is False
    assert ok.tightened is False


def test_champion_blind_spot_detected_on_val():
    """The seed champion-0 has a reward-hacking blind spot: hack ratio must dip
    below the exploitation threshold on the gamed val anchors."""
    champion = versioning.get_champion_rubric_text()
    controller = rqgm_adapter.get_controller()
    rep = controller.assess(champion, anchor_ds.load_anchors(anchor_ds.VAL), persist=False)
    assert rep.mean_hack_ratio is not None
    assert rep.mean_hack_ratio < 1.0
    assert rep.exploitation_detected is True


def test_gate_config_is_configurable():
    champion = versioning.get_champion_rubric_text()
    better = mutate_rubric_text(
        champion,
        [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "challenger-cfg",
        1,
    )
    strict = gate.GateConfig(bootstrap_iters=500, bootstrap_alpha=0.05)
    res = gate.evaluate_promotion(champion, better, domain_id="smart_manufacturing", epoch=0, config=strict)
    assert res.passed is True
    assert res.bootstrap_alpha == 0.05


# ---------------------------------------------------------------------------
# Selective erasure: soft-delete + reconfirm (physics_truth preserved forever)
# ---------------------------------------------------------------------------
def test_approve_advances_epoch_and_selective_erasure():
    mem = EvolutionaryMemory(collection="test_evolve_erasure")
    # A lesson the promoted champion (reward_hacking_resistance) STILL endorses:
    mem.add(
        "gamed the KPI by disabling the sensor",
        MemoryType.HEURISTIC_FAILURE,
        created_at_epoch=0,
        extra={"depends_on_evaluator_judgement": True, "reconfirm_flaws": ["kpi_sensor_gaming"]},
    )
    # A lesson keyed on a flaw the new champion does NOT cover -> obsolete -> soft-deleted:
    mem.add(
        "never recalibrated after a supplier change",
        MemoryType.HEURISTIC_FAILURE,
        created_at_epoch=0,
        extra={"depends_on_evaluator_judgement": True, "reconfirm_flaws": ["concept_drift_blind"]},
    )
    mem.add("valve stiction causes flow/command divergence", MemoryType.PHYSICS_TRUTH, created_at_epoch=0)

    prop = evolve.propose_challenger()  # adds reward_hacking_resistance
    assert versioning.get_epoch() == 0

    result = evolve.approve_challenger(prop.version, approve=True, memory=mem)
    assert result["applied"] is True
    assert result["epoch_id"] == 1
    assert versioning.get_epoch() == 1
    # One reconfirmed (kpi still caught), one soft-deleted (drift not caught).
    assert result["reconfirmed_memories"] == 1
    assert result["erased_memories"] == 1

    stats = mem.stats()
    assert stats["physics_truth"] == 1  # physics never erased
    # Soft-delete is reversible: the row still exists, just inactive.
    assert mem.count_by_type(MemoryType.HEURISTIC_FAILURE) == 2
    assert mem.count_by_type(MemoryType.HEURISTIC_FAILURE, active_only=True) == 1


def test_reject_does_not_advance_epoch():
    prop = evolve.propose_challenger()
    assert versioning.get_epoch() == 0
    result = evolve.approve_challenger(prop.version, approve=False)
    assert result["applied"] is False
    assert versioning.get_epoch() == 0


def test_promoted_champion_contains_evolved_criterion():
    prop = evolve.propose_challenger()
    evolve.approve_challenger(prop.version, approve=True, memory=EvolutionaryMemory(collection="t2"))
    champion = versioning.get_champion_rubric_text()
    assert 'origin="gepa"' in champion
