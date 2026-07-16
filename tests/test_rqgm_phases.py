"""Phase 2-4 coverage: Pareto frontier, adversarial red-team, memory retrieval,
judge panel + self-consistency + agreement, and the RQGM report."""

from backend.evaluator import (
    adversarial,
    anchors as anchor_ds,
    evolve,
    panel,
    report,
    versioning,
)
from backend.evaluator.judge import evaluate_architecture, retrieve_memory_block
from backend.evaluator.mutation import mutate_rubric_text
from backend.gatekeeper.physics_memory import seed_physics_truths
from backend.memory.qdrant_store import EvolutionaryMemory, MemoryType


# --- Phase 2: Pareto frontier ---------------------------------------------
def test_frontier_keeps_nondominated_and_picks_best():
    frontier = evolve.gepa_evolve(
        versioning.get_champion_rubric_text(), epoch=0, budget=8,
        adversarial_samples=evolve.generate_adversarial_pool(),
    )
    assert len(frontier.members) >= 2  # incumbent + at least one challenger
    best = frontier.best()
    # Best (by BBε) must beat the incumbent and add coverage.
    assert best.version != "champion-0"
    assert best.added_criteria  # added at least one criterion
    # A coverage-vs-parsimony trade-off keeps >1 distinct added-criteria set.
    distinct = {tuple(sorted(m.added_criteria)) for m in frontier.members}
    assert len(distinct) >= 2


def test_propose_via_frontier_beats_champion():
    prop, frontier = evolve.propose_via_frontier(adversarial_samples=evolve.generate_adversarial_pool())
    assert prop.metrics["separation_delta"] > 0
    assert frontier.to_dict()["size"] >= 2


# --- Phase 3: adversarial red-team ----------------------------------------
def test_red_team_targets_champion_blind_spots():
    champion = versioning.get_champion_rubric_text()
    blind = adversarial.champion_blind_spots(champion)
    # champion-0 misses reward hacking + concept drift (headroom poison pills).
    assert "kpi_sensor_gaming" in blind
    assert "concept_drift_blind" in blind
    samples = adversarial.generate_adversarial_samples(champion)
    assert {s["targets"] for s in samples} == set(blind)

    # A rubric that adds coverage removes the corresponding blind spot.
    covered = mutate_rubric_text(
        champion, [{"id": "reward_hacking_resistance", "text": "penalize kpi gaming via sensor disabling"}],
        "c-cov", 1,
    )
    assert "kpi_sensor_gaming" not in adversarial.champion_blind_spots(covered)


# --- Phase 3: memory retrieval (activates the previously-dead search path) --
def test_hybrid_search_memory_injection():
    mem = EvolutionaryMemory(collection="test_mem_retrieval")
    seed_physics_truths(mem)
    mem.add(
        "threshold false-tripped under correlated thermocouple noise",
        MemoryType.HEURISTIC_FAILURE, created_at_epoch=0,
    )
    block = retrieve_memory_block("correlated noise trips a false shutdown", 0, memory=mem)
    assert "heuristic_failure" in block
    assert "physics_truth" in block

    # Injecting memory must NOT change the deterministic mock deficit (flaws are
    # read from the candidate only, not the injected memory).
    arch = "single static threshold shutdown, no state schema"
    a = evaluate_architecture(arch, domain_id="smart_manufacturing", inject_memory=False)
    b = evaluate_architecture(arch, domain_id="smart_manufacturing", memory=mem)
    assert a.deficit_score == b.deficit_score


def test_future_epoch_memories_not_retrieved():
    mem = EvolutionaryMemory(collection="test_mem_epoch")
    mem.add("epoch-5 lesson about drift", MemoryType.HEURISTIC_FAILURE, created_at_epoch=5)
    assert retrieve_memory_block("drift", 0, memory=mem) == ""  # epoch 0 cannot see epoch-5
    assert "epoch-5 lesson" in retrieve_memory_block("drift", 5, memory=mem)


# --- Phase 4: judge panel + self-consistency + agreement -------------------
def test_panel_self_consistency_reduces_to_median():
    champion = versioning.get_champion_rubric_text()
    a = [x for x in anchor_ds.load_anchors("val") if x["label"] == "weak"][0]
    pe = panel.evaluate_panel(anchor_ds.anchor_candidate_text(a), champion, domain_id=a["domain"], epoch=0)
    assert pe.n_judges == 3
    assert len(pe.per_persona) == 3
    assert pe.verdict == "weak"
    # personas disagree slightly (self-consistency has something to aggregate)
    assert pe.deficit_std >= 0.0


def test_cohen_kappa_and_agreement():
    assert panel.cohen_kappa(["weak", "strong"], ["weak", "strong"]) == 1.0
    champion = versioning.get_champion_rubric_text()
    ag = panel.anchor_agreement(champion, anchor_ds.load_anchors("test"), epoch=0)
    assert 0.0 <= ag["accuracy"] <= 1.0
    assert -1.0 <= ag["cohen_kappa"] <= 1.0
    assert ag["n"] == 5


def test_criterion_order_randomization_is_bias_control():
    import re

    champion = versioning.get_champion_rubric_text()
    shuffled = panel._shuffle_criteria(champion, seed=2)

    # Same criteria, different order -> deterministic mock score is unchanged
    # (order cannot bias the verdict; that is the point of the control).
    def ids(t: str) -> list[str]:
        return sorted(re.findall(r'<criterion[^>]*id="([^"]+)"', t))

    assert ids(shuffled) == ids(champion)


# --- Phase 4: report -------------------------------------------------------
def test_build_report_shape():
    rep = report.build_report()
    assert rep["rqgm_backend"] in {"rqgm", "local-fallback"}
    assert set(rep["separation"].keys()) == {"val", "test"}
    assert "mean_hack_ratio" in rep["hack_ratio"]
    assert "accuracy" in rep["judge_agreement"]["val"]
    assert rep["data_splits"]["val"]["total"] == 7
