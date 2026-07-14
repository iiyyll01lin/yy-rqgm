"""Shared pytest fixtures.

Force the inference MOCK so the whole suite runs with no live Lemonade/GPU/net,
and reset RQGM epoch state around each test for isolation.
"""

import os

os.environ.setdefault("LEMONADE_FORCE_MOCK", "1")

import pytest  # noqa: E402

from backend.evaluator import versioning  # noqa: E402


@pytest.fixture(autouse=True)
def _clean_epoch_state():
    versioning.reset()
    yield
    versioning.reset()
