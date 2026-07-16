"""Shared pytest fixtures.

Force the inference MOCK so the whole suite runs with no live Lemonade/GPU/net,
and reset RQGM epoch state around each test for isolation.

``LEMONADE_FORCE_MOCK=1`` is set unconditionally so EVERY non-live test stays
offline/deterministic even during a live run. Tests marked ``@pytest.mark.live``
opt into a real model by constructing their own ``force_mock=False`` client; they
are SKIPPED by default (see :func:`pytest_collection_modifyitems`) since there is
no GPU/model here.
"""

import os

os.environ.setdefault("LEMONADE_FORCE_MOCK", "1")

import pytest  # noqa: E402

from backend.evaluator import gate, report, rqgm_adapter, versioning  # noqa: E402

_RUN_LIVE = os.getenv("RQGM_RUN_LIVE", "").lower() in ("1", "true", "yes")


def pytest_collection_modifyitems(config, items):
    """Skip ``@pytest.mark.live`` tests unless ``RQGM_RUN_LIVE`` is set.

    They require a real local model (Lemonade / vLLM-ROCm) + GPU. Skipping (rather
    than deselecting) keeps them visible as a SKIPPED count in the default run.
    """
    if _RUN_LIVE:
        return
    skip_live = pytest.mark.skip(
        reason="live model test; set RQGM_RUN_LIVE=1 and LEMONADE_BASE_URL to run"
    )
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(autouse=True)
def _clean_epoch_state():
    versioning.reset()
    rqgm_adapter.get_controller().reset()
    gate.reset_sequential_state()
    report.reset_metrics_ledger()
    yield
    versioning.reset()
    rqgm_adapter.get_controller().reset()
    gate.reset_sequential_state()
    report.reset_metrics_ledger()
