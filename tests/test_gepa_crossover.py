"""p2-gepa-plus (optional): GEPA System-Aware Merge (crossover) — native adoption.

The official ``gepa`` PyPI engine's System-Aware Merge is not installable in this
offline env (no wheel; adding it would break the no-network default path), so the
crossover CONCEPT is adopted natively: union two frontier members' complementary
GEPA-added criteria into a broader-coverage child. Opt-in; the gate is unaffected.
"""

from backend.evaluator import evolve, gate, versioning
from backend.evaluator.frontier import FrontierMember
from backend.evaluator.mutation import mutate_rubric_text
from backend.inference.mock_scoring import rubric_criteria_ids
from backend.memory.qdrant_store import EvolutionaryMemory

_DOMAIN = "smart_manufacturing"


def test_extract_gepa_criteria_round_trips():
    champ = versioning.get_champion_rubric_text()
    text = mutate_rubric_text(
        champ, [{"id": "reward_hacking_resistance", "text": "penalize kpi gaming via sensor disabling"}],
        "c-x", 1,
    )
    crit = evolve._extract_gepa_criteria(text)
    assert [c["id"] for c in crit] == ["reward_hacking_resistance"]
    assert "kpi gaming" in crit[0]["text"]


def test_system_aware_merge_unions_complementary_criteria_and_separates_better():
    champ = versioning.get_champion_rubric_text()
    a_text = mutate_rubric_text(champ, [{"id": "reward_hacking_resistance", "text": "penalize kpi gaming"}], "cha", 1)
    b_text = mutate_rubric_text(champ, [{"id": "drift_monitoring", "text": "penalize missing drift monitoring"}], "chb", 1)
    a = FrontierMember(version="cha", rubric_text=a_text, objectives={}, bbe=0.0, added_criteria=["reward_hacking_resistance"])
    b = FrontierMember(version="chb", rubric_text=b_text, objectives={}, bbe=0.0, added_criteria=["drift_monitoring"])

    child_text, added = evolve.system_aware_merge(a, b, champ, "child-xover", 1)
    assert set(added) == {"reward_hacking_resistance", "drift_monitoring"}
    assert {"reward_hacking_resistance", "drift_monitoring"} <= rubric_criteria_ids(child_text)

    # The merged child closes BOTH blind spots, so it separates at least as well as
    # either single-criterion parent on the held-out val split, and passes the gate.
    res_a = gate.evaluate_promotion(champ, a_text, domain_id=_DOMAIN, epoch=0)
    res_child = gate.evaluate_promotion(champ, child_text, domain_id=_DOMAIN, epoch=0)
    assert res_child.separation_delta >= res_a.separation_delta
    assert res_child.n_wins >= res_a.n_wins
    assert res_child.passed is True


def test_gepa_evolve_with_crossover_runs_and_still_gates():
    fr = evolve.gepa_evolve(
        versioning.get_champion_rubric_text(), epoch=0, budget=6,
        adversarial_samples=evolve.generate_adversarial_pool(), crossover=True,
    )
    best = fr.best()
    assert best is not None and best.added_criteria

    prop, _fr = evolve.propose_via_frontier(
        adversarial_samples=evolve.generate_adversarial_pool(), crossover=True
    )
    res = evolve.approve_challenger(
        prop.version, approve=True, memory=EvolutionaryMemory(collection="t_xover")
    )
    assert res["gate"]["passed"] is True
    assert res["applied"] is True
