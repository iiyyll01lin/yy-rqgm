"""Tier feasibility — the decision half of the Static Gatekeeper.

Combines the pure VRAM + bandwidth math with a tier's capacity/bandwidth to
answer the questions the wizard and the LangGraph orchestrator need:

    * Does the workload physically fit? (hard gate — physics never evolves)
    * How much headroom is left?
    * Roughly how fast will it decode?
    * How large a population/concurrency fits under prefix caching?

Everything is deterministic and side-effect free.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from backend.gatekeeper import vram
from backend.gatekeeper.bandwidth import DEFAULT_EFFICIENCY, tokens_per_s
from backend.gatekeeper.spec import ModelSpec, TierSpec


@dataclass(frozen=True)
class Feasibility:
    """Result of evaluating one workload against one tier (or custom spec)."""

    feasible: bool
    vram_total_gb: float
    breakdown: vram.VramBreakdown
    headroom_gb: float
    tokens_per_s_est: float
    max_population: int
    kv_savings_from_prefix_pct: float

    def to_dict(self) -> dict:
        return {
            "feasible": self.feasible,
            "vram_total_gb": self.vram_total_gb,
            "vram_breakdown": self.breakdown.to_dict(),
            "tokens_per_s_est": self.tokens_per_s_est,
            "headroom_gb": self.headroom_gb,
            "max_population": self.max_population,
            "kv_savings_from_prefix_pct": self.kv_savings_from_prefix_pct,
        }


def max_population(
    memory_gb: float,
    model: ModelSpec,
    seq_len: int,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
) -> int:
    """Max number of population/concurrency branches that fit in ``memory_gb``.

    Solves ``memory >= overhead + (weights + shared_prefix + P*branch)*(1+act)``
    for the largest integer P, consistent with :func:`vram.vram_breakdown`'s
    activation model. Returns 0 when even a single branch does not fit.
    """
    resolved_dtype = dtype or model.dtype_default
    weights = vram.weights_gb(model.params_b, resolved_dtype)
    overhead = vram.overhead_gb()
    per_tok_gb = (
        vram.kv_bytes_per_token(model.n_layers, model.n_kv_heads, model.head_dim)
        / vram.GB
    )
    prefix_ratio = min(max(prefix_ratio, 0.0), 1.0)
    prefix_len = int(seq_len * prefix_ratio)
    branch_len = seq_len - prefix_len
    shared_prefix_gb = per_tok_gb * prefix_len
    per_branch_gb = per_tok_gb * branch_len

    act = 1.0 + vram.ACTIVATION_FRACTION
    fixed = overhead + (weights + shared_prefix_gb) * act
    per_branch_eff = per_branch_gb * act

    available = memory_gb - fixed
    if available < 0:
        return 0
    if per_branch_eff <= 0:
        # Fully-shared prefix (branch_len == 0): capacity is not branch-limited.
        return 1_000_000
    return int(math.floor(available / per_branch_eff))


def _evaluate(
    memory_gb: float,
    bandwidth_tbs: float,
    model: ModelSpec,
    seq_len: int,
    concurrency: int = 1,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> Feasibility:
    resolved_dtype = dtype or model.dtype_default
    concurrency = max(int(concurrency), 1)

    breakdown = vram.vram_breakdown(
        model,
        seq_len=seq_len,
        concurrency=concurrency,
        dtype=resolved_dtype,
        prefix_ratio=prefix_ratio,
    )
    total = breakdown.total
    headroom = memory_gb - total

    # Single-stream decode estimate: read all weights + this context's KV per step.
    weights = vram.weights_gb(model.params_b, resolved_dtype)
    kv_read = vram.kv_cache_gb(
        model.n_layers, model.n_kv_heads, model.head_dim, seq_len=seq_len, batch=1
    )
    tps = tokens_per_s(
        bandwidth_tbs=bandwidth_tbs,
        weights_gb=weights,
        kv_read_gb=kv_read,
        efficiency=efficiency,
    )

    return Feasibility(
        feasible=total <= memory_gb,
        vram_total_gb=total,
        breakdown=breakdown,
        headroom_gb=headroom,
        tokens_per_s_est=tps,
        max_population=max_population(
            memory_gb, model, seq_len, dtype=resolved_dtype, prefix_ratio=prefix_ratio
        ),
        kv_savings_from_prefix_pct=vram.prefix_savings_pct(
            seq_len=seq_len, population=concurrency, prefix_ratio=prefix_ratio
        ),
    )


def evaluate_tier(
    tier: TierSpec,
    model: ModelSpec,
    seq_len: int,
    concurrency: int = 1,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> Feasibility:
    """Evaluate a workload against a hardware tier from the DB."""
    return _evaluate(
        memory_gb=tier.memory_gb,
        bandwidth_tbs=tier.bandwidth_tbs,
        model=model,
        seq_len=seq_len,
        concurrency=concurrency,
        dtype=dtype,
        prefix_ratio=prefix_ratio,
        efficiency=efficiency,
    )


def evaluate_custom(
    memory_gb: float,
    bandwidth_tbs: float,
    model: ModelSpec,
    seq_len: int,
    concurrency: int = 1,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
    efficiency: float = DEFAULT_EFFICIENCY,
) -> Feasibility:
    """Evaluate a workload against an ad-hoc (user-supplied) hardware spec."""
    return _evaluate(
        memory_gb=memory_gb,
        bandwidth_tbs=bandwidth_tbs,
        model=model,
        seq_len=seq_len,
        concurrency=concurrency,
        dtype=dtype,
        prefix_ratio=prefix_ratio,
        efficiency=efficiency,
    )
