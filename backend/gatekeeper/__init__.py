"""Static Hardware Gatekeeper — DETERMINISTIC, never evolves.

This package is the *trust foundation* of the platform: pure, importable,
side-effect-free physics. Given a model, a sequence length and a concurrency,
it computes VRAM (weights + KV cache + activations + overhead) and a
memory-bandwidth-bound tokens/s estimate, then decides tier feasibility.

Physics does not evolve — so nothing in here ever calls an LLM, reads mutable
state, or depends on the RQGM evaluator. See ``backend.evaluator`` for the part
that *does* evolve.
"""

from backend.gatekeeper.spec import (  # noqa: F401
    ModelSpec,
    TierSpec,
    list_models,
    list_tiers,
    get_model,
    get_tier,
)

__all__ = [
    "ModelSpec",
    "TierSpec",
    "list_models",
    "list_tiers",
    "get_model",
    "get_tier",
]
