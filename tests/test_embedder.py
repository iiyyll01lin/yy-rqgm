"""p0-embedder: pluggable embedder seam.

The default (and CI) path MUST stay the deterministic hashing bag-of-words so the
suite is offline/reproducible; a real semantic embedder is opt-in via
``AGENTFORGE_EMBEDDER`` and falls back to hashing on any unavailability. These
tests prove the seam + fallback WITHOUT needing sentence-transformers or a server.
"""

import hashlib
import math

import numpy as np

from backend.memory.qdrant_store import (
    EMBED_DIM,
    EvolutionaryMemory,
    HashingEmbedder,
    MemoryType,
    embed,
    get_embedder,
)


def test_hashing_embed_is_deterministic_and_l2_normalized():
    v1 = embed("correlated sensor noise trips a false shutdown")
    v2 = embed("correlated sensor noise trips a false shutdown")
    assert v1 == v2  # deterministic (no RNG/network)
    assert len(v1) == EMBED_DIM
    assert math.isclose(math.sqrt(sum(x * x for x in v1)), 1.0, abs_tol=1e-5)


def test_default_embedder_is_hashing_and_matches_module_embed(monkeypatch):
    monkeypatch.delenv("AGENTFORGE_EMBEDDER", raising=False)
    emb = get_embedder()
    assert isinstance(emb, HashingEmbedder)
    assert emb.dim == EMBED_DIM
    # The object wrapper is byte-identical to the module-level function (router
    # and the store therefore keep their exact current behaviour offline).
    assert emb.embed("valve stiction") == embed("valve stiction", EMBED_DIM)


def test_optin_but_unavailable_backend_falls_back_to_hashing(monkeypatch):
    # sentence-transformers is NOT installed in CI: opting in must NOT raise and
    # must NOT touch the network — it silently returns the deterministic default.
    monkeypatch.setenv("AGENTFORGE_EMBEDDER", "sentence-transformers")
    assert isinstance(get_embedder(), HashingEmbedder)
    # A served-model embedder with the inference mock forced is likewise unavailable.
    monkeypatch.setenv("AGENTFORGE_EMBEDDER", "lemonade")
    assert isinstance(get_embedder(), HashingEmbedder)
    # An unknown backend id is also a safe fallback.
    monkeypatch.setenv("AGENTFORGE_EMBEDDER", "totally-bogus")
    assert isinstance(get_embedder(), HashingEmbedder)


class _FakeEmbedder:
    """A tiny 8-dim deterministic embedder proving the store is embedder-agnostic."""

    name = "fake-test-embedder"
    dim = 8

    def embed(self, text: str) -> list[float]:
        digest = hashlib.sha256(text.encode("utf-8")).digest()
        vec = np.frombuffer(digest[:8], dtype=np.uint8).astype(np.float32)
        norm = float(np.linalg.norm(vec)) or 1.0
        return (vec / norm).tolist()


def test_store_uses_injected_embedder_for_dim_and_vectors():
    mem = EvolutionaryMemory(collection="test_injected_embedder", embedder=_FakeEmbedder())
    assert mem.dim == 8  # collection sized to the (real) embedder, not EMBED_DIM
    mem.add("valve stiction root-cause fact", MemoryType.PHYSICS_TRUTH, created_at_epoch=0)
    mem.add("threshold false-trip heuristic", MemoryType.HEURISTIC_FAILURE, created_at_epoch=0)
    hits = mem.search("valve stiction", top_k=2, memory_type=MemoryType.PHYSICS_TRUTH)
    assert hits and "valve stiction" in hits[0].text

    stats = mem.stats()
    assert stats["embedder"] == "fake-test-embedder"
    assert stats["embedding_dim"] == 8


def test_default_store_reports_hashing_embedder_in_stats():
    mem = EvolutionaryMemory(collection="test_default_stats")
    stats = mem.stats()
    assert stats["embedder"].startswith("hashing-bow")
    assert stats["embedding_dim"] == EMBED_DIM
