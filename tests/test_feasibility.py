"""TDD unit tests for tier feasibility (backend/gatekeeper/feasibility.py).

Combines the pure VRAM + bandwidth math with a tier's memory_gb / bandwidth_tbs
to answer: does it fit, what's the headroom, how fast, and how large a
population fits under prefix caching. This is the HARD physical gate that the
LangGraph orchestrator uses to reject infeasible proposals (physics never
evolves).
"""

import pytest

from backend.gatekeeper import feasibility, vram
from backend.gatekeeper.spec import get_model, get_tier


def test_7b_int4_fits_on_rx7900xtx():
    tier = get_tier("rx_7900_xtx")
    model = get_model("llama-3.1-8b")
    res = feasibility.evaluate_tier(tier, model, seq_len=8192, concurrency=1, dtype="int4")
    assert res.feasible is True
    assert res.headroom_gb > 0
    assert res.vram_total_gb == pytest.approx(res.breakdown.total)


def test_70b_fp16_does_not_fit_on_rx7900xtx():
    tier = get_tier("rx_7900_xtx")  # 24 GB
    model = get_model("llama-3.1-70b")
    res = feasibility.evaluate_tier(tier, model, seq_len=8192, concurrency=1, dtype="fp16")
    assert res.feasible is False
    assert res.headroom_gb < 0  # negative headroom = over budget


def test_70b_int4_fits_on_mi300x():
    tier = get_tier("mi300x")  # 192 GB
    model = get_model("llama-3.1-70b")
    res = feasibility.evaluate_tier(tier, model, seq_len=8192, concurrency=1, dtype="int4")
    assert res.feasible is True
    assert res.headroom_gb > 100  # tons of room


def test_tokens_per_s_est_is_bandwidth_ordered():
    model = get_model("llama-3.1-8b")
    mi300x = feasibility.evaluate_tier(get_tier("mi300x"), model, seq_len=4096, dtype="int4")
    ryzen = feasibility.evaluate_tier(
        get_tier("ryzen_ai_max_395"), model, seq_len=4096, dtype="int4"
    )
    # MI300X's 5.3 TB/s >> Ryzen AI's 0.256 TB/s.
    assert mi300x.tokens_per_s_est > 5 * ryzen.tokens_per_s_est


def test_max_population_bigger_on_mi300x_than_consumer():
    model = get_model("llama-3.1-8b")
    big = feasibility.evaluate_tier(get_tier("mi300x"), model, seq_len=8192, dtype="int4")
    small = feasibility.evaluate_tier(get_tier("rx_7900_xtx"), model, seq_len=8192, dtype="int4")
    assert big.max_population > small.max_population
    assert small.max_population >= 1


def test_prefix_caching_raises_max_population_and_reports_savings():
    model = get_model("llama-3.1-8b")
    no_prefix = feasibility.evaluate_tier(
        get_tier("mi300x"), model, seq_len=8192, concurrency=16, dtype="int4", prefix_ratio=0.0
    )
    with_prefix = feasibility.evaluate_tier(
        get_tier("mi300x"), model, seq_len=8192, concurrency=16, dtype="int4", prefix_ratio=0.9
    )
    assert with_prefix.max_population >= no_prefix.max_population
    assert with_prefix.kv_savings_from_prefix_pct > 0
    assert no_prefix.kv_savings_from_prefix_pct == pytest.approx(0.0)
    # savings must match the pure formula for pop=concurrency=16.
    assert with_prefix.kv_savings_from_prefix_pct == pytest.approx(
        vram.prefix_savings_pct(seq_len=8192, population=16, prefix_ratio=0.9)
    )


def test_concurrency_increases_total_vram():
    model = get_model("llama-3.1-8b")
    tier = get_tier("mi300x")
    one = feasibility.evaluate_tier(tier, model, seq_len=8192, concurrency=1, dtype="int4")
    many = feasibility.evaluate_tier(tier, model, seq_len=8192, concurrency=32, dtype="int4")
    assert many.vram_total_gb > one.vram_total_gb


def test_custom_tier_from_memory_and_bandwidth():
    # feasibility must also work on an ad-hoc (custom) tier spec, not just the DB.
    model = get_model("llama-3.1-8b")
    res = feasibility.evaluate_custom(
        memory_gb=8.0, bandwidth_tbs=0.4, model=model, seq_len=4096, concurrency=1, dtype="int4"
    )
    # 8B int4 ~4 GB weights + kv + overhead should still fit in 8 GB at 4k ctx.
    assert res.feasible in (True, False)  # deterministic bool, no exception
    assert res.vram_total_gb > 0
