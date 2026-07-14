"""Evolutionary memory (Qdrant) with selective erasure.

Stores two kinds of memory with an epoch tag so RQGM can *selectively erase*
obsolete negative results on epoch upgrade while preserving physical truths:

    * ``heuristic_failure`` — fuzzy negative results tied to an evaluator epoch
      (safe to erase when that epoch is superseded)
    * ``physics_truth``     — deterministic facts from the gatekeeper (NEVER erased)

Degrades gracefully to Qdrant's local in-memory mode when no server is running.
"""

from backend.memory.qdrant_store import (  # noqa: F401
    EvolutionaryMemory,
    MemoryType,
    get_memory,
)

__all__ = ["EvolutionaryMemory", "MemoryType", "get_memory"]
