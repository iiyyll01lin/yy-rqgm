"""Phase 5 (p2_search): Thompson-sampling frontier, hardened adversary, and the
over-acceptance / over-optimization report monitors."""

import random

from backend.evaluator import adversarial, evolve, report, versioning
from backend.evaluator.frontier import FrontierMember, ParetoFrontier
from backend.inference.mock_scoring import FLAW_CATALOG


# --- Thompson-sampling parent selection -----------------------------------
def test_thompson_sampling_prefers_productive_parent():
    fr = ParetoFrontier(top_k=8)
    a = FrontierMember(version="a", rubric_text="", objectives={"x": 1.0}, bbe=0.1)
    b = FrontierMember(version="b", rubric_text="", objectives={"y": 1.0}, bbe=0.1)
    fr.members = [a, b]
    # 'a' consistently parents gate-improving children; 'b' never does.
    for _ in range(20):
        fr.record_child_outcome(a, improved=True)
        fr.record_child_outcome(b, improved=False)
    assert a.child_trials == 20 and a.child_successes == 20
    assert b.child_trials == 20 and b.child_successes == 0

    rng = random.Random(0)
    picks = [fr.sample_stochastic(rng).version for _ in range(300)]
    # Thompson sampling exploits the parent with the better posterior.
    assert picks.count("a") > picks.count("b")


def test_frontier_exposes_bandit_state_and_is_updated():
    frontier = evolve.gepa_evolve(
        versioning.get_champion_rubric_text(), epoch=0, budget=4,
        adversarial_samples=evolve.generate_adversarial_pool(),
    )
    d = frontier.to_dict()
    assert d["members"]
    assert all("child_trials" in m and "child_successes" in m for m in d["members"])
    # the incumbent (max parsimony, never dominated) parented children in the loop
    assert sum(m["child_trials"] for m in d["members"]) >= 1


# --- Hardened adversary ----------------------------------------------------
def test_adversarial_disguises_stacked_and_out_of_catalog():
    champ = versioning.get_champion_rubric_text()
    blind = adversarial.champion_blind_spots(champ)
    samples = adversarial.generate_adversarial_samples(champ)

    # Varied disguises: more than one distinct strength profile is used.
    profiles = {tuple(s["strengths"]) for s in samples}
    assert len(profiles) >= 2
    # Stacked poison pills: at least one sample carries two flaws.
    assert any(len(s["flaws"]) >= 2 for s in samples)
    # Default targets still equal the champion blind spots (no id drift).
    assert {s["targets"] for s in samples} == set(blind)

    # Out-of-catalog gaming is opt-in and introduces flaws the catalog cannot map.
    ooc = [
        s for s in adversarial.generate_adversarial_samples(champ, include_out_of_catalog=True)
        if s["id"].startswith("adv_ooc_")
    ]
    assert ooc
    assert all(flaw not in FLAW_CATALOG for s in ooc for flaw in s["flaws"])


# --- Report monitors -------------------------------------------------------
def test_report_surfaces_over_acceptance_and_over_optimization():
    rep = report.build_report(include_agreement=False)

    oa = rep["over_acceptance"]
    assert 0.0 <= oa["over_acceptance_rate"] <= 1.0
    assert oa["n"] >= 1 and "per_sample" in oa
    assert oa["accepted_as_strong"] <= oa["n"]

    oo = rep["over_optimization"]
    assert set(oo) == {"proxy_val_separation", "gold_test_separation", "separation_gap"}
    assert oo["separation_gap"] == round(oo["proxy_val_separation"] - oo["gold_test_separation"], 4)

    # Existing keys the frontend/tests read must be untouched (additive only).
    assert set(rep["separation"].keys()) == {"val", "test"}
    assert "mean_hack_ratio" in rep["hack_ratio"]

    hs = report.health_summary()
    assert 0.0 <= hs["over_acceptance_rate"] <= 1.0
    assert "proxy_gold_separation_gap" in hs
    assert "val_separation" in hs and "test_separation" in hs
