"""p0-anchors: the second domain pack (grid_energy) + larger anchor set.

Proves the RQGM loop generalizes to a SECOND domain and that expanding the
held-out anchor set did not break the code gate's promotion behaviour.
"""

from backend.domains.registry import get_domain, list_domains
from backend.evaluator import anchors as anchor_ds
from backend.evaluator import gate, versioning
from backend.evaluator.judge import score_candidate
from backend.evaluator.mutation import mutate_rubric_text
from backend.graph.router import best_template

_GRID = "grid_energy"


# --- discovery / pack contents ---------------------------------------------
def test_grid_energy_domain_is_discovered_and_complete():
    assert _GRID in [d.id for d in list_domains()]
    pack = get_domain(_GRID)
    assert pack is not None
    assert len(pack.poison_pills()) >= 3
    template_ids = {t.id for t in pack.workflow_templates()}
    assert {"frequency_regulation", "der_dispatch", "microgrid_islanding"} <= template_ids
    frag = pack.rubric_fragment()
    for cid in ("grid_frequency_regulation", "islanding_safety", "der_forecast_resilience"):
        assert f'id="{cid}"' in frag


def test_router_maps_grid_needs_to_templates():
    assert best_template("hold grid frequency in the regulation band via AGC balancing", _GRID) == "frequency_regulation"
    assert best_template("economic dispatch of solar and battery storage against forecasts", _GRID) == "der_dispatch"
    assert best_template("microgrid islanding and breaker reclose reconnection", _GRID) == "microgrid_islanding"


# --- grid anchors are scorable offline (deterministic mock) ----------------
def test_grid_anchors_are_scorable_offline():
    champion = versioning.get_champion_rubric_text()
    grid = [a for a in anchor_ds.load_all_anchors() if a.get("domain") == _GRID]
    assert len(grid) >= 20  # a substantive second-domain pack
    for a in grid:
        deficit = score_candidate(
            anchor_ds.anchor_candidate_text(a), champion, domain_id=_GRID, epoch=0
        )["deficit_loose"]
        # The champion (merged with the grid fragment) separates the planted
        # ground truth: weak anchors score >= tau, strong anchors well below it.
        if a["label"] == "weak":
            assert deficit >= 0.30, f"{a['id']} weak but scored {deficit}"
        else:
            assert deficit < 0.30, f"{a['id']} strong but scored {deficit}"


# --- the gate still promotes correctly on the larger multi-domain val set ---
def test_gate_still_promotes_reward_hacking_on_multidomain_val():
    champion = versioning.get_champion_rubric_text()
    better = mutate_rubric_text(
        champion,
        [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "challenger-multidomain",
        1,
    )
    res = gate.evaluate_promotion(champion, better, epoch=0)
    assert res.passed is True
    assert res.effect_size >= res.min_detectable_effect
    assert res.posterior_prob_improvement >= 0.95
    # Wins come from BOTH domains' kpi_sensor_gaming val-weak anchors (5 sm + 2 grid).
    assert res.n_wins == 7
    assert res.per_flaw_wins["kpi_sensor_gaming"]["win"] == 7
    assert res.n_val == 19
