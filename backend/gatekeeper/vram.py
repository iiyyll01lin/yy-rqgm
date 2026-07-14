"""Deterministic VRAM math — the physics half of the Static Gatekeeper.

Pure functions only. No I/O, no globals, no LLM. Everything here is exact and
testable; see ``tests/test_vram.py`` for the hand-computed ground truth.

Core formulas (shared with the blueprint):

    M_weights   = N_params * bytes_per_param          # int4=0.5, fp8=1, fp16=2 B/param
    KV_per_tok  = 2 * n_layers * n_kv_heads * head_dim * bytes_kv
    M_kv        = KV_per_tok * seq_len * batch
    # prefix caching (population / MCTS branches):
    #   effective_KV = shared_prefix_KV + P * branch_KV   (replaces P * full_KV)
    VRAM_total  = M_weights + M_kv + M_act + framework_overhead

Unit convention: **GB decimal** (1 GB = 1e9 bytes).
"""

from __future__ import annotations

from dataclasses import dataclass

from backend.gatekeeper.spec import ModelSpec

# 1 GB = 1e9 bytes (decimal). Chosen so hand-computed cases stay clean
# (7B int4 == 3.5 GB exactly).
GB: float = 1_000_000_000.0

# Bytes per stored *weight* parameter, keyed by (case-insensitive) dtype.
BYTES_PER_PARAM: dict[str, float] = {
    "int4": 0.5,
    "int8": 1.0,
    "fp8": 1.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
}

# Bytes per stored *KV-cache* element. KV is usually kept at fp16 even when
# weights are int4-quantized (some stacks use fp8 KV to save memory).
KV_DTYPE_BYTES: dict[str, float] = {
    "int4": 0.5,
    "int8": 1.0,
    "fp8": 1.0,
    "fp16": 2.0,
    "bf16": 2.0,
    "fp32": 4.0,
}

# Transient inference activation / working-buffer estimate, expressed as a
# fraction of (weights + KV). Decode activations are small relative to weights
# and KV; this is a deliberately simple, documented heuristic (not exact
# physics) so the breakdown always has a defensible activations line item.
ACTIVATION_FRACTION: float = 0.10

# Fixed framework / runtime reservation (allocator, fragmentation, HIP/CUDA-like
# context, CUDA graphs, scheduler buffers). A flat constant keeps it deterministic.
FRAMEWORK_OVERHEAD_GB: float = 1.0


def bytes_per_param(dtype: str) -> float:
    """Bytes per stored weight for a dtype. Case-insensitive. Raises KeyError."""
    return BYTES_PER_PARAM[dtype.lower()]


def _kv_bytes(kv_dtype: str) -> float:
    return KV_DTYPE_BYTES[kv_dtype.lower()]


def weights_gb(params_b: float, dtype: str) -> float:
    """Model weight memory in GB. ``params_b`` is parameter count in billions."""
    return params_b * 1e9 * bytes_per_param(dtype) / GB


def kv_bytes_per_token(
    n_layers: int, n_kv_heads: int, head_dim: int, kv_dtype: str = "fp16"
) -> float:
    """Bytes of KV cache consumed per token. Factor 2 accounts for K *and* V."""
    return 2.0 * n_layers * n_kv_heads * head_dim * _kv_bytes(kv_dtype)


def kv_cache_gb(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    batch: int = 1,
    kv_dtype: str = "fp16",
) -> float:
    """KV-cache memory in GB for ``batch`` independent sequences of ``seq_len``."""
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_dtype)
    return per_tok * seq_len * batch / GB


def _prefix_split(seq_len: int, prefix_ratio: float) -> tuple[int, int]:
    """Return (shared_prefix_len, per_branch_len). Prefix len is floored."""
    prefix_ratio = min(max(prefix_ratio, 0.0), 1.0)
    prefix_len = int(seq_len * prefix_ratio)
    branch_len = seq_len - prefix_len
    return prefix_len, branch_len


def kv_cache_gb_with_prefix(
    n_layers: int,
    n_kv_heads: int,
    head_dim: int,
    seq_len: int,
    population: int,
    prefix_ratio: float = 0.0,
    kv_dtype: str = "fp16",
) -> float:
    """KV-cache memory in GB for a *population* of branches sharing a prefix.

    Under vLLM-style prefix caching the shared prompt is stored once, and only
    each branch's divergent suffix is duplicated:

        effective_KV = shared_prefix_KV + population * branch_KV

    With ``prefix_ratio == 0`` this collapses to the naive ``population * full``.
    """
    per_tok = kv_bytes_per_token(n_layers, n_kv_heads, head_dim, kv_dtype)
    prefix_len, branch_len = _prefix_split(seq_len, prefix_ratio)
    effective_tokens = prefix_len + population * branch_len
    return per_tok * effective_tokens / GB


def prefix_savings_pct(seq_len: int, population: int, prefix_ratio: float) -> float:
    """Percent KV saved by prefix caching vs. the naive population*full baseline."""
    prefix_len, branch_len = _prefix_split(seq_len, prefix_ratio)
    effective_tokens = prefix_len + population * branch_len
    naive_tokens = seq_len * population
    if naive_tokens <= 0:
        return 0.0
    return (1.0 - effective_tokens / naive_tokens) * 100.0


def activations_gb(
    weights_gb_val: float, kv_gb_val: float, fraction: float = ACTIVATION_FRACTION
) -> float:
    """Heuristic transient activation memory as a fraction of weights + KV."""
    return fraction * (weights_gb_val + kv_gb_val)


def overhead_gb() -> float:
    """Fixed framework/runtime VRAM reservation."""
    return FRAMEWORK_OVERHEAD_GB


@dataclass(frozen=True)
class VramBreakdown:
    """The four VRAM line items surfaced by the API contract."""

    weights: float
    kv_cache: float
    activations: float
    overhead: float

    @property
    def total(self) -> float:
        return self.weights + self.kv_cache + self.activations + self.overhead

    def to_dict(self) -> dict:
        return {
            "weights": self.weights,
            "kv_cache": self.kv_cache,
            "activations": self.activations,
            "overhead": self.overhead,
        }


def vram_breakdown(
    model: ModelSpec,
    seq_len: int,
    concurrency: int = 1,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
    kv_dtype: str = "fp16",
) -> VramBreakdown:
    """Full VRAM breakdown for a model at a given seq_len / concurrency.

    ``concurrency`` is the batch / population dimension (concurrent requests or
    MCTS/population branches). ``prefix_ratio`` enables prefix-cache sharing.
    ``dtype`` defaults to the model's catalog default when None.
    """
    resolved_dtype = dtype or model.dtype_default
    w = weights_gb(model.params_b, resolved_dtype)
    kv = kv_cache_gb_with_prefix(
        n_layers=model.n_layers,
        n_kv_heads=model.n_kv_heads,
        head_dim=model.head_dim,
        seq_len=seq_len,
        population=max(concurrency, 1),
        prefix_ratio=prefix_ratio,
        kv_dtype=kv_dtype,
    )
    act = activations_gb(w, kv)
    ov = overhead_gb()
    return VramBreakdown(weights=w, kv_cache=kv, activations=act, overhead=ov)


def vram_total_gb(
    model: ModelSpec,
    seq_len: int,
    concurrency: int = 1,
    dtype: str | None = None,
    prefix_ratio: float = 0.0,
    kv_dtype: str = "fp16",
) -> float:
    """Convenience: total VRAM in GB."""
    return vram_breakdown(
        model, seq_len, concurrency, dtype, prefix_ratio, kv_dtype
    ).total
