"""Tests for the RQGM/GEPA evolution loop + versioning + selective erasure."""

from backend.evaluator import evolve, versioning
from backend.memory.qdrant_store import EvolutionaryMemory, MemoryType


def test_propose_challenger_registers_and_scores():
    prop = evolve.propose_challenger()
    assert prop.version.startswith("challenger-")
    assert prop.new_criteria  # always yields a concrete change
    assert "champion_separation" in prop.metrics
    assert "challenger_separation" in prop.metrics
    assert prop.rubric_diff  # a non-empty unified diff
    # registered in versioning
    assert versioning.get_challenger(prop.version) is not None


def test_approve_advances_epoch_and_selective_erasure():
    mem = EvolutionaryMemory(collection="test_evolve_erasure")
    mem.add("epoch0 heuristic A", MemoryType.HEURISTIC_FAILURE, created_at_epoch=0)
    mem.add("epoch0 heuristic B", MemoryType.HEURISTIC_FAILURE, created_at_epoch=0)
    mem.add("physics truth", MemoryType.PHYSICS_TRUTH, created_at_epoch=0)

    prop = evolve.propose_challenger()
    assert versioning.get_epoch() == 0

    result = evolve.approve_challenger(prop.version, approve=True, memory=mem)
    assert result["applied"] is True
    assert result["epoch_id"] == 1
    assert versioning.get_epoch() == 1
    assert result["erased_memories"] == 2  # both epoch-0 heuristics erased

    stats = mem.stats()
    assert stats["heuristic_failure"] == 0
    assert stats["physics_truth"] == 1  # physics never erased


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
