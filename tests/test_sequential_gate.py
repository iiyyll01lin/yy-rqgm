"""p1-sequential-gate: cross-epoch anytime-valid (e-process) family-wise control.

The per-epoch P1+P2 gate is unchanged (default config). When
``GateConfig.sequential_correction`` is enabled, a persisted e-process (a test
martingale of per-epoch Bayes-factor e-values) additionally guards against
family-wise error from REPEATED promotions: promoting on noise can never
accumulate enough wealth to keep clearing ``1/alpha``.
"""

import math

from backend.evaluator import evolve, gate, versioning
from backend.evaluator.gate import GateConfig, epoch_e_value, evaluate_promotion
from backend.evaluator.mutation import mutate_rubric_text
from backend.memory.qdrant_store import EvolutionaryMemory

_DOMAIN = "smart_manufacturing"


def _reward_hacking_fix() -> tuple[str, str]:
    champion = versioning.get_champion_rubric_text()
    better = mutate_rubric_text(
        champion,
        [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "challenger-seq",
        1,
    )
    return champion, better


# --- e-value primitive -----------------------------------------------------
def test_epoch_e_value_properties():
    assert epoch_e_value(0, 0) == 1.0                       # no directional evidence
    assert epoch_e_value(7, 0) > epoch_e_value(4, 0) > 1.0  # more wins => more evidence
    assert epoch_e_value(0, 7) < 1.0                        # losses => evidence FOR the null
    # One-sided Bayes factor for 7-0 (uniform prior on θ∈(0.5,1)): 255/8 = 31.875.
    assert math.isclose(epoch_e_value(7, 0), 31.875, rel_tol=1e-9)


# --- default behaviour is unchanged (per-epoch P1+P2 only) -----------------
def test_default_config_leaves_gate_unchanged_but_reports_e_process():
    champion, better = _reward_hacking_fix()
    res = evaluate_promotion(champion, better, domain_id=_DOMAIN, epoch=0)
    assert res.passed is True                    # per-epoch gate promotes as before
    assert res.sequential_correction is False    # correction is OFF by default
    # e-process fields are still reported (advisory) for transparency.
    assert res.sequential_e_value == 31.875
    assert res.sequential_looks_prior == 0


# --- enabling the correction: lenient budget lets a strong first look pass --
def test_sequential_correction_passes_strong_first_look():
    champion, better = _reward_hacking_fix()
    cfg = GateConfig(sequential_correction=True, family_wise_alpha=0.1)  # threshold 1/0.1 = 10
    res = evaluate_promotion(champion, better, domain_id=_DOMAIN, epoch=0, config=cfg)
    assert res.sequential_passed is True         # wealth 16 >= 10
    assert res.passed is True
    assert res.sequential_threshold == 10.0


# --- strict budget blocks a single look that P1+P2 accept (family-wise) ----
def test_sequential_correction_blocks_underpowered_single_look():
    champion, better = _reward_hacking_fix()
    cfg = GateConfig(sequential_correction=True, family_wise_alpha=0.001)  # threshold 1000
    res = evaluate_promotion(champion, better, domain_id=_DOMAIN, epoch=0, config=cfg)
    # The per-epoch signal is decisive...
    assert res.p1_non_inferior is True
    assert res.p2_passed is True
    # ...but a single look's e-value (16) has not cleared the family-wise bar.
    assert res.sequential_passed is False
    assert res.passed is False
    assert "P3 sequential" in res.reason


# --- genuine evidence ACCUMULATES across looks and unlocks promotion -------
def test_sequential_wealth_accumulates_across_committed_looks():
    champion, better = _reward_hacking_fix()
    cfg = GateConfig(sequential_correction=True, family_wise_alpha=0.001)  # threshold 1000
    gate.reset_sequential_state()
    res1 = evaluate_promotion(champion, better, domain_id=_DOMAIN, epoch=0, config=cfg)
    assert res1.passed is False  # 16 < 1000

    gate.commit_sequential_look(res1.sequential_e_value, epoch=1)  # wealth 16
    gate.commit_sequential_look(res1.sequential_e_value, epoch=2)  # wealth 256
    res2 = evaluate_promotion(champion, better, domain_id=_DOMAIN, epoch=3, config=cfg)
    assert res2.sequential_looks_prior == 2
    assert res2.sequential_passed is True  # 256 * 16 = 4096 >= 1000
    assert res2.passed is True


# --- promoting on NOISE never accumulates wealth (the whole point) ---------
def test_noise_promotions_do_not_accumulate_wealth():
    gate.reset_sequential_state()
    for _ in range(20):
        gate.commit_sequential_look(1.0)  # e-value ~1 => no evidence
    state = gate.load_sequential_state()
    assert abs(state["log_wealth"]) < 1e-9  # wealth stayed at 1
    assert state["looks"] == 20


# --- integration: the approve path spends one look per applied promotion ---
def test_approve_with_sequential_correction_commits_one_look():
    gate.reset_sequential_state()
    cfg = GateConfig(sequential_correction=True, family_wise_alpha=0.1)
    prop = evolve.propose_challenger()  # adds reward_hacking_resistance
    res = evolve.approve_challenger(
        prop.version, approve=True, gate_config=cfg,
        memory=EvolutionaryMemory(collection="t_seq_commit"),
    )
    assert res["applied"] is True
    assert res["gate"]["sequential_correction"] is True
    assert res["gate"]["sequential_passed"] is True
    state = gate.load_sequential_state()
    assert state["looks"] == 1  # exactly one promotion spent one look of the budget
