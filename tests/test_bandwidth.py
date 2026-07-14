"""TDD unit tests for the deterministic bandwidth -> tokens/s math.

Decode is memory-bandwidth bound: each generated token must stream the model
weights (and the KV cache for the active context) out of memory. So the upper
bound is roughly bandwidth / bytes_read_per_token.
"""

import pytest

from backend.gatekeeper import bandwidth


def test_tokens_per_s_weight_bound_mi300x():
    # MI300X: 5.3 TB/s. 7B int4 weights = 3.5 GB.
    # 5.3e12 / 3.5e9 = 1514.2857... tok/s (weight-bound upper bound).
    got = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5)
    assert got == pytest.approx(5.3e12 / 3.5e9)
    assert got == pytest.approx(1514.2857, abs=0.5)


def test_tokens_per_s_ryzen_ai_is_much_slower():
    # Ryzen AI Max+ 395: 0.256 TB/s -> same model ~ 73 tok/s (memory-bound).
    got = bandwidth.tokens_per_s(bandwidth_tbs=0.256, weights_gb=3.5)
    assert got == pytest.approx(256.0 / 3.5, abs=0.5)  # 73.14


def test_efficiency_scales_linearly():
    full = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5, efficiency=1.0)
    half = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5, efficiency=0.5)
    assert half == pytest.approx(0.5 * full)


def test_kv_read_term_reduces_throughput():
    without_kv = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5, kv_read_gb=0.0)
    with_kv = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5, kv_read_gb=1.5)
    assert with_kv < without_kv
    assert with_kv == pytest.approx(5.3e12 / ((3.5 + 1.5) * 1e9))


def test_zero_bytes_guard_returns_zero():
    # Degenerate input must not raise ZeroDivisionError.
    assert bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=0.0, kv_read_gb=0.0) == 0.0


def test_higher_bandwidth_gives_more_tokens():
    slow = bandwidth.tokens_per_s(bandwidth_tbs=0.864, weights_gb=3.5)
    fast = bandwidth.tokens_per_s(bandwidth_tbs=5.3, weights_gb=3.5)
    assert fast > slow
