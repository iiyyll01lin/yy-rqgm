"""Qdrant-backed evolutionary memory with hybrid search + selective erasure.

Payload schema per point::

    {
      "text": str,
      "memory_type": "heuristic_failure" | "physics_truth",
      "created_at_epoch": int,
      "active": bool,
      "created_at": int (epoch seconds),
      ... arbitrary extra ...
    }

Connection strategy (graceful degradation):
    * If ``QDRANT_URL`` is set and reachable -> use that server.
    * Otherwise -> Qdrant *local mode* (:memory:), an embedded pure-python store
      with the same filtering semantics. So the platform runs with no server.

Embeddings are pluggable (see :func:`get_embedder`). The DEFAULT is a lightweight,
deterministic hashing bag-of-words (numpy only) — no torch / sentence-transformers,
so CI stays fully offline and reproducible. The ``AGENTFORGE_EMBEDDER`` env toggle
opts in to a REAL local semantic embedder — ``sentence-transformers`` on CPU
(``AGENTFORGE_EMBEDDER=sentence-transformers``) or a served model's OpenAI-compatible
``/v1/embeddings`` endpoint (``=lemonade``) — WHEN it is installed / reachable, with
automatic fallback to the deterministic hashing embed otherwise. The store API is
embedder-agnostic: only :func:`get_embedder` changes.
"""

from __future__ import annotations

import hashlib
import os
import time
import uuid
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from qdrant_client import QdrantClient
from qdrant_client import models as qm

EMBED_DIM = 256
DEFAULT_COLLECTION = "agentforge_memory"
_CONNECT_TIMEOUT_S = 1.5

# Embedder selection (offline-safe): the default is the deterministic hashing
# bag-of-words; the env toggle opts in to a real local embedder WHEN available.
ENV_EMBEDDER = "AGENTFORGE_EMBEDDER"          # hashing (default) | sentence-transformers | lemonade
ENV_EMBED_MODEL = "AGENTFORGE_EMBED_MODEL"    # embedder-specific model id (optional)
_HASHING = "hashing"
_SENTENCE_TRANSFORMERS = "sentence-transformers"
_LEMONADE = "lemonade"
# CPU-friendly, widely-cached small model; only downloaded if a user opts in.
_DEFAULT_ST_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_EMBED_TIMEOUT_S = 30.0


class MemoryType(str, Enum):
    HEURISTIC_FAILURE = "heuristic_failure"
    PHYSICS_TRUTH = "physics_truth"


@dataclass
class MemoryHit:
    id: str
    text: str
    memory_type: str
    created_at_epoch: int
    active: bool
    score: float
    payload: dict[str, Any]

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "text": self.text,
            "memory_type": self.memory_type,
            "created_at_epoch": self.created_at_epoch,
            "active": self.active,
            "score": self.score,
        }


def embed(text: str, dim: int = EMBED_DIM) -> list[float]:
    """Deterministic hashing bag-of-words embedding, L2-normalized.

    Not a transformer — a cheap, offline, dependency-light stand-in that still
    gives useful lexical-semantic similarity for the PoC. It stays the DEFAULT
    (and the fallback) so the test suite is offline/deterministic; it also powers
    :mod:`backend.graph.router`'s keyword+embedding cosine.

    The live-recall SEAM is now implemented (see :func:`get_embedder`): set
    ``AGENTFORGE_EMBEDDER=sentence-transformers`` (or ``=lemonade``) to swap in a
    real semantic embedder WHEN it is installed/reachable, with automatic fallback
    to this hashing embed. The store API is unchanged — only the embedder object
    differs.
    """
    vec = np.zeros(dim, dtype=np.float32)
    tokens = [t for t in _tokenize(text)]
    for tok in tokens:
        h = int(hashlib.md5(tok.encode("utf-8")).hexdigest(), 16)
        idx = h % dim
        sign = 1.0 if (h // dim) % 2 == 0 else -1.0
        vec[idx] += sign
    norm = float(np.linalg.norm(vec))
    if norm > 0:
        vec /= norm
    else:
        # avoid all-zero vectors (cosine undefined); nudge one dim.
        vec[0] = 1.0
    return vec.tolist()


def _tokenize(text: str) -> list[str]:
    return [t for t in "".join(c.lower() if c.isalnum() else " " for c in text).split() if t]


# ---------------------------------------------------------------------------
# Pluggable embedders (default = deterministic hashing; real ones are opt-in)
# ---------------------------------------------------------------------------
class HashingEmbedder:
    """Deterministic hashing bag-of-words embedder (the offline default).

    Thin object wrapper over :func:`embed` so the store can be embedder-agnostic
    while the default path stays byte-identical to the previous behaviour.
    """

    def __init__(self, dim: int = EMBED_DIM) -> None:
        self.dim = int(dim)
        self.name = f"hashing-bow[{self.dim}]"

    def embed(self, text: str) -> list[float]:
        return embed(text, self.dim)


class SentenceTransformerEmbedder:
    """Real local semantic embedder via ``sentence-transformers`` (CPU, offline).

    OPTIONAL dependency — install with ``uv pip install sentence-transformers``
    (not in the default lock, so CI never pulls torch/transformers). Construction
    raises if the package or model is unavailable so :func:`get_embedder` can fall
    back to the deterministic hashing embed.
    """

    def __init__(self, model_name: str | None = None) -> None:
        from sentence_transformers import SentenceTransformer  # optional import

        self.model_name = model_name or os.getenv(ENV_EMBED_MODEL, _DEFAULT_ST_MODEL)
        self._model = SentenceTransformer(self.model_name, device="cpu")
        self.dim = int(self._model.get_sentence_embedding_dimension())
        self.name = f"sentence-transformers:{self.model_name}"

    def embed(self, text: str) -> list[float]:
        vec = self._model.encode(
            [text], normalize_embeddings=True, convert_to_numpy=True, show_progress_bar=False
        )[0]
        return np.asarray(vec, dtype=np.float32).tolist()


class LemonadeEmbedder:
    """Real embedder via a served model's OpenAI-compatible ``/v1/embeddings``.

    Uses the local ROCm inference server (Lemonade / vLLM-ROCm) — the truly
    "local semantic recall" path the docs promise. Construction probes the
    endpoint once and raises on any failure (no server, mock, unreachable) so
    :func:`get_embedder` falls back to the deterministic hashing embed. Never
    used by the default offline test suite (which forces the mock).
    """

    def __init__(self, model_name: str | None = None) -> None:
        from backend.inference.lemonade_client import get_lemonade_client

        self._client = get_lemonade_client()
        self.model_name = model_name or os.getenv(ENV_EMBED_MODEL) or None
        probe = self._client.embeddings(["probe"], model=self.model_name)
        if not probe or not probe[0]:
            raise RuntimeError("live /v1/embeddings unavailable (mock/offline/unreachable)")
        self.dim = len(probe[0])
        self.name = f"lemonade:{self.model_name or self._client.model}"

    def embed(self, text: str) -> list[float]:
        out = self._client.embeddings([text], model=self.model_name)
        if not out or not out[0]:
            # Defensive: degrade to a deterministic hashing vector of the same dim
            # rather than raising mid-request (keeps the pipeline alive).
            return embed(text, self.dim)
        return [float(x) for x in out[0]]


def get_embedder(dim: int = EMBED_DIM):
    """Return the active embedder, honouring ``AGENTFORGE_EMBEDDER`` (offline-safe).

    Default (unset / ``hashing``) → deterministic :class:`HashingEmbedder`. Opting
    in to a real embedder (``sentence-transformers`` / ``lemonade``) falls back to
    the hashing embedder on ANY unavailability, so the default test suite (which
    never sets the env, and forces the inference mock) stays offline + deterministic.
    """
    backend = os.getenv(ENV_EMBEDDER, _HASHING).strip().lower()
    if backend in ("", _HASHING, "bow", "hashing-bow"):
        return HashingEmbedder(dim)
    try:
        if backend in (_SENTENCE_TRANSFORMERS, "st", "sbert"):
            return SentenceTransformerEmbedder()
        if backend in (_LEMONADE, "vllm", "server"):
            return LemonadeEmbedder()
    except Exception:
        # Availability gate: unknown/uninstalled/unreachable → deterministic fallback.
        return HashingEmbedder(dim)
    return HashingEmbedder(dim)


class EvolutionaryMemory:
    """Hybrid (semantic + payload-filter) memory with soft-delete + purge."""

    def __init__(
        self,
        url: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        dim: int = EMBED_DIM,
        embedder: Any = None,
    ):
        # The embedder owns the vector space; the collection is sized to its dim so
        # a real (e.g. 384-d sentence-transformers) embedder just works. Default =
        # deterministic hashing (offline). ``embedder`` can be injected (tests).
        self._embedder = embedder if embedder is not None else get_embedder(dim=dim)
        self.collection = collection
        self.dim = int(getattr(self._embedder, "dim", dim))
        self.mode = "local"
        self.client = self._connect(url)
        self._ensure_collection()

    def _embed(self, text: str) -> list[float]:
        return self._embedder.embed(text)

    def _connect(self, url: str | None) -> QdrantClient:
        url = url or os.getenv("QDRANT_URL")
        if url:
            try:
                client = QdrantClient(url=url, timeout=_CONNECT_TIMEOUT_S)
                client.get_collections()  # probe
                self.mode = "server"
                return client
            except Exception:
                # Server configured but unreachable -> fall back silently.
                pass
        self.mode = "local"
        return QdrantClient(location=":memory:")

    def _ensure_collection(self) -> None:
        try:
            if not self.client.collection_exists(self.collection):
                self.client.create_collection(
                    collection_name=self.collection,
                    vectors_config=qm.VectorParams(size=self.dim, distance=qm.Distance.COSINE),
                )
        except Exception:
            # Best-effort: some client versions raise if it already exists.
            pass

    # -- write -------------------------------------------------------------
    def add(
        self,
        text: str,
        memory_type: MemoryType | str,
        created_at_epoch: int = 0,
        active: bool = True,
        extra: dict[str, Any] | None = None,
    ) -> str:
        mt = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
        point_id = str(uuid.uuid4())
        payload: dict[str, Any] = {
            "text": text,
            "memory_type": mt,
            "created_at_epoch": int(created_at_epoch),
            "active": bool(active),
            "created_at": int(time.time()),
        }
        if extra:
            payload.update(extra)
        self.client.upsert(
            collection_name=self.collection,
            points=[qm.PointStruct(id=point_id, vector=self._embed(text), payload=payload)],
        )
        return point_id

    # -- read (hybrid: semantic + filter) ----------------------------------
    def search(
        self,
        query: str,
        top_k: int = 5,
        memory_type: MemoryType | str | None = None,
        active_only: bool = True,
        max_epoch: int | None = None,
    ) -> list[MemoryHit]:
        conditions: list[qm.FieldCondition] = []
        if memory_type is not None:
            mt = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
            conditions.append(qm.FieldCondition(key="memory_type", match=qm.MatchValue(value=mt)))
        if active_only:
            conditions.append(qm.FieldCondition(key="active", match=qm.MatchValue(value=True)))
        if max_epoch is not None:
            conditions.append(
                qm.FieldCondition(key="created_at_epoch", range=qm.Range(lte=max_epoch))
            )
        query_filter = qm.Filter(must=conditions) if conditions else None

        result = self.client.query_points(
            collection_name=self.collection,
            query=self._embed(query),
            query_filter=query_filter,
            limit=top_k,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for p in result.points:
            payload = p.payload or {}
            hits.append(
                MemoryHit(
                    id=str(p.id),
                    text=str(payload.get("text", "")),
                    memory_type=str(payload.get("memory_type", "")),
                    created_at_epoch=int(payload.get("created_at_epoch", 0)),
                    active=bool(payload.get("active", True)),
                    score=float(p.score) if p.score is not None else 0.0,
                    payload=payload,
                )
            )
        return hits

    # -- read (non-semantic scan for selective erasure) --------------------
    def fetch(
        self,
        memory_type: MemoryType | str | None = None,
        max_epoch: int | None = None,
        active_only: bool = True,
        limit: int = 10000,
    ) -> list[MemoryHit]:
        """Scan (no vector query) memories matching payload filters.

        Used by selective erasure to enumerate candidate ``heuristic_failure``
        memories for reconfirmation. ``physics_truth`` is simply never requested.
        """
        conditions: list[qm.FieldCondition] = []
        if memory_type is not None:
            mt = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
            conditions.append(qm.FieldCondition(key="memory_type", match=qm.MatchValue(value=mt)))
        if active_only:
            conditions.append(qm.FieldCondition(key="active", match=qm.MatchValue(value=True)))
        if max_epoch is not None:
            conditions.append(qm.FieldCondition(key="created_at_epoch", range=qm.Range(lte=max_epoch)))
        query_filter = qm.Filter(must=conditions) if conditions else None
        points, _ = self.client.scroll(
            collection_name=self.collection,
            scroll_filter=query_filter,
            limit=limit,
            with_payload=True,
        )
        hits: list[MemoryHit] = []
        for p in points:
            payload = p.payload or {}
            hits.append(
                MemoryHit(
                    id=str(p.id),
                    text=str(payload.get("text", "")),
                    memory_type=str(payload.get("memory_type", "")),
                    created_at_epoch=int(payload.get("created_at_epoch", 0)),
                    active=bool(payload.get("active", True)),
                    score=0.0,
                    payload=payload,
                )
            )
        return hits

    # -- soft delete / selective erasure -----------------------------------
    def soft_delete(self, ids: list[str]) -> int:
        """Mark points inactive (reversible)."""
        if not ids:
            return 0
        self.client.set_payload(
            collection_name=self.collection,
            payload={"active": False},
            points=list(ids),
        )
        return len(ids)

    def mark_reconfirmed(self, ids: list[str], epoch: int) -> int:
        """Stamp ``reconfirmed_epoch`` on memories the new champion still endorses."""
        if not ids:
            return 0
        self.client.set_payload(
            collection_name=self.collection,
            payload={"reconfirmed_epoch": int(epoch)},
            points=list(ids),
        )
        return len(ids)

    def purge_epoch(
        self,
        up_to_epoch: int,
        memory_type: MemoryType | str = MemoryType.HEURISTIC_FAILURE,
    ) -> int:
        """Selective erasure: hard-delete memories of one type at/below an epoch.

        ``physics_truth`` is never purged (it does not evolve). This is the
        RQGM selective-erasure step run when an epoch is superseded.
        """
        mt = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
        # count first (for reporting)
        flt = qm.Filter(
            must=[
                qm.FieldCondition(key="memory_type", match=qm.MatchValue(value=mt)),
                qm.FieldCondition(key="created_at_epoch", range=qm.Range(lte=up_to_epoch)),
            ]
        )
        n = self.count(count_filter=flt)
        self.client.delete(
            collection_name=self.collection,
            points_selector=qm.FilterSelector(filter=flt),
        )
        return n

    # -- utils -------------------------------------------------------------
    def count(self, count_filter: qm.Filter | None = None) -> int:
        try:
            return int(
                self.client.count(
                    collection_name=self.collection,
                    count_filter=count_filter,
                    exact=True,
                ).count
            )
        except Exception:
            return 0

    def count_by_type(self, memory_type: MemoryType | str, active_only: bool = False) -> int:
        mt = memory_type.value if isinstance(memory_type, MemoryType) else str(memory_type)
        conds = [qm.FieldCondition(key="memory_type", match=qm.MatchValue(value=mt))]
        if active_only:
            conds.append(qm.FieldCondition(key="active", match=qm.MatchValue(value=True)))
        return self.count(qm.Filter(must=conds))

    def stats(self) -> dict[str, Any]:
        return {
            "mode": self.mode,
            "collection": self.collection,
            "embedder": getattr(self._embedder, "name", "unknown"),
            "embedding_dim": self.dim,
            "total": self.count(),
            "heuristic_failure": self.count_by_type(MemoryType.HEURISTIC_FAILURE),
            "physics_truth": self.count_by_type(MemoryType.PHYSICS_TRUTH),
        }


# ---------------------------------------------------------------------------
# Module-level singleton
# ---------------------------------------------------------------------------
_memory: EvolutionaryMemory | None = None


def get_memory() -> EvolutionaryMemory:
    global _memory
    if _memory is None:
        _memory = EvolutionaryMemory()
    return _memory
