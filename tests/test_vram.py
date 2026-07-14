"""TDD unit tests for the deterministic VRAM math (backend/gatekeeper/vram.py).

These encode *hand-computed* ground truth. They are the trust foundation of the
platform and must never be loosened to make code pass — the physics does not
evolve. Unit convention: GB decimal (1 GB = 1e9 bytes).
"""

import math

import pytest

from backend.gatekeeper import vram
from backend.gatekeeper.spec import ModelSpec


# --- weights ---------------------------------------------------------------
def test_bytes_per_param_table():
    assert vram.bytes_per_param("int4") == 0.5
    assert vram.bytes_per_param("fp8") == 1.0
    assert vram.bytes_per_param("int8") == 1.0
    assert vram.bytes_per_param("fp16") == 2.0
    assert vram.bytes_per_param("bf16") == 2.0
    assert vram.bytes_per_param("fp32") == 4.0


def test_bytes_per_param_is_case_insensitive():
    assert vram.bytes_per_param("INT4") == 0.5
    assert vram.bytes_per_param("FP16") == 2.0


def test_bytes_per_param_unknown_raises():
    with pytest.raises(KeyError):
        vram.bytes_per_param("int3")


def test_weights_7b_int4_is_3point5_gb():
    # The canonical hand-computed case from the spec: 7B int4 ~= 3.5 GB.
    assert vram.weights_gb(7.0, "int4") == pytest.approx(3.5)


def test_weights_scaling_by_dtype():
    assert vram.weights_gb(7.0, "fp16") == pytest.approx(14.0)
    assert vram.weights_gb(7.0, "fp8") == pytest.approx(7.0)
    assert vram.weights_gb(70.0, "int4") == pytest.approx(35.0)


# --- KV cache --------------------------------------------------------------
def test_kv_bytes_per_token_hand_computed():
    # Llama-3-8B-ish GQA dims: 2 (K,V) * n_layers * n_kv_heads * head_dim * bytes.
    # 2 * 32 * 8 * 128 * 2 = 131072 bytes/token.
    assert (
        vram.kv_bytes_per_token(n_layers=32, n_kv_heads=8, head_dim=128, kv_dtype="fp16")
        == 131072
    )


def test_kv_bytes_per_token_fp8_halves():
    assert (
        vram.kv_bytes_per_token(n_layers=32, n_kv_heads=8, head_dim=128, kv_dtype="fp8")
        == 65536
    )


def test_kv_cache_gb_single_stream_8192_ctx():
    # 131072 B/tok * 8192 tok / 1e9 = 1.073741824 GB (== exactly 1 GiB).
    got = vram.kv_cache_gb(
        n_layers=32, n_kv_heads=8, head_dim=128, seq_len=8192, batch=1, kv_dtype="fp16"
    )
    assert got == pytest.approx(1.073741824)


def test_kv_cache_gb_scales_linearly_with_batch():
    single = vram.kv_cache_gb(
        n_layers=32, n_kv_heads=8, head_dim=128, seq_len=8192, batch=1
    )
    ten = vram.kv_cache_gb(
        n_layers=32, n_kv_heads=8, head_dim=128, seq_len=8192, batch=10
    )
    assert ten == pytest.approx(10 * single)


# --- prefix caching (population / MCTS branches) ---------------------------
def test_prefix_caching_ratio_zero_equals_naive():
    naive = vram.kv_cache_gb(
        n_layers=32, n_kv_heads=8, head_dim=128, seq_len=1000, batch=10
    )
    prefixed = vram.kv_cache_gb_with_prefix(
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        seq_len=1000,
        population=10,
        prefix_ratio=0.0,
    )
    assert prefixed == pytest.approx(naive)


def test_prefix_caching_savings_hand_computed():
    # seq=1000, pop=10, 90% shared prefix.
    # effective tokens = prefix(900) + pop*branch(10*100) = 1900
    # naive tokens = seq*pop = 10000  ->  savings = 1 - 1900/10000 = 81%.
    assert (
        vram.prefix_savings_pct(seq_len=1000, population=10, prefix_ratio=0.9)
        == pytest.approx(81.0)
    )


def test_prefix_caching_reduces_kv():
    prefixed = vram.kv_cache_gb_with_prefix(
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        seq_len=1000,
        population=10,
        prefix_ratio=0.9,
    )
    # 131072 * 1900 / 1e9
    assert prefixed == pytest.approx(131072 * 1900 / 1e9)


def test_prefix_savings_single_member_is_zero():
    # With population=1 there is nothing to share across, savings ~ 0.
    assert vram.prefix_savings_pct(seq_len=1000, population=1, prefix_ratio=0.9) == pytest.approx(
        0.0
    )


# --- full breakdown --------------------------------------------------------
def _demo_model() -> ModelSpec:
    return ModelSpec(
        id="demo-7b",
        name="Demo 7B",
        params_b=7.0,
        n_layers=32,
        n_kv_heads=8,
        head_dim=128,
        hidden=4096,
        context_len=8192,
        dtype_default="int4",
    )


def test_vram_breakdown_components_and_total():
    m = _demo_model()
    b = vram.vram_breakdown(m, seq_len=8192, concurrency=1, dtype="int4")
    assert b.weights == pytest.approx(3.5)
    assert b.kv_cache == pytest.approx(1.073741824)
    # activations = fraction * (weights + kv); overhead = fixed framework reserve.
    assert b.activations == pytest.approx(
        vram.ACTIVATION_FRACTION * (b.weights + b.kv_cache)
    )
    assert b.overhead == pytest.approx(vram.FRAMEWORK_OVERHEAD_GB)
    assert b.total == pytest.approx(b.weights + b.kv_cache + b.activations + b.overhead)


def test_vram_breakdown_defaults_dtype_from_model():
    m = _demo_model()
    b_explicit = vram.vram_breakdown(m, seq_len=8192, concurrency=1, dtype="int4")
    b_default = vram.vram_breakdown(m, seq_len=8192, concurrency=1, dtype=None)
    assert b_default.weights == pytest.approx(b_explicit.weights)


def test_vram_total_is_positive_and_finite():
    m = _demo_model()
    total = vram.vram_total_gb(m, seq_len=4096, concurrency=4, dtype="fp16")
    assert total > 0
    assert math.isfinite(total)
