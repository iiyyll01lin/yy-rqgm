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

Embeddings are a lightweight, deterministic hashing bag-of-words (numpy only) —
no torch / sentence-transformers, so it works fully offline. Swap in a real
embedder later without changing the store API.
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
    gives useful lexical-semantic similarity for the PoC.
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


class EvolutionaryMemory:
    """Hybrid (semantic + payload-filter) memory with soft-delete + purge."""

    def __init__(
        self,
        url: str | None = None,
        collection: str = DEFAULT_COLLECTION,
        dim: int = EMBED_DIM,
    ):
        self.collection = collection
        self.dim = dim
        self.mode = "local"
        self.client = self._connect(url)
        self._ensure_collection()

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
            points=[qm.PointStruct(id=point_id, vector=embed(text, self.dim), payload=payload)],
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
            query=embed(query, self.dim),
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
