"""Deterministic memory-bandwidth -> tokens/s math.

LLM *decode* is memory-bandwidth bound: to emit each new token the accelerator
must stream the model weights (and, more precisely, the KV cache for the active
context) out of HBM/GDDR/LPDDR. So the throughput upper bound is roughly:

    tokens_per_s ~= efficiency * mem_bandwidth / bytes_read_per_token

This is why a 128 GB Ryzen AI part (~0.256 TB/s) can *hold* a big model but
decodes far slower than a 192 GB MI300X (~5.3 TB/s). Pure + testable; see
``tests/test_bandwidth.py``.
"""

from __future__ import annotations

from backend.gatekeeper.vram import GB

# Fraction of theoretical memory bandwidth realistically achievable during
# decode (kernel overheads, non-contiguous KV reads, etc.). The API surfaces
# estimates with this applied; the pure formula defaults to 1.0 (upper bound).
DEFAULT_EFFICIENCY: float = 0.7

TB_PER_S: float = 1e12  # 1 TB/s = 1e12 bytes/s


def tokens_per_s(
    bandwidth_tbs: float,
    weights_gb: float,
    kv_read_gb: float = 0.0,
    efficiency: float = 1.0,
) -> float:
    """Estimate single-stream decode tokens/s from memory bandwidth.

    ``bandwidth_tbs``  memory bandwidth in TB/s.
    ``weights_gb``     bytes (in GB) of weights read per decode step.
    ``kv_read_gb``     bytes (in GB) of KV cache read per decode step (context).
    ``efficiency``     achievable fraction of peak bandwidth (0..1].

    Returns 0.0 for degenerate (zero-byte) inputs rather than raising.
    """
    bytes_per_token = (weights_gb + kv_read_gb) * GB
    if bytes_per_token <= 0:
        return 0.0
    bandwidth_bytes_per_s = bandwidth_tbs * TB_PER_S
    return efficiency * bandwidth_bytes_per_s / bytes_per_token
