"""Tests for the Lemonade client mock fallback + JSON parsing."""

import json

from backend.inference.lemonade_client import LemonadeClient, MockMarker
from backend.inference.parsing import extract_json


def test_force_mock_client_is_not_live():
    c = LemonadeClient(force_mock=True)
    assert c.is_live() is False
    assert c.using_mock is True


def test_unreachable_endpoint_falls_back_to_mock():
    # Point at a definitely-dead port; must not raise, must return a string.
    c = LemonadeClient(base_url="http://127.0.0.1:59999/api/v1", force_mock=False)
    out = c.chat("hello")
    assert isinstance(out, str)
    assert out  # non-empty deterministic stub
    assert c.using_mock is True


def test_mock_is_deterministic():
    c = LemonadeClient(force_mock=True)
    a = c.chat([{"role": "user", "content": "same input"}])
    b = c.chat([{"role": "user", "content": "same input"}])
    assert a == b


def test_task_agent_marker_returns_valid_json():
    c = LemonadeClient(force_mock=True)
    out = c.chat([{"role": "system", "content": MockMarker.TASK_AGENT.value}, {"role": "user", "content": "pdm"}])
    parsed = extract_json(out)
    assert parsed is not None
    assert "architecture" in parsed


def test_evaluator_marker_returns_scored_json():
    c = LemonadeClient(force_mock=True)
    out = c.chat(
        [
            {"role": "system", "content": MockMarker.EVALUATOR.value},
            {"role": "user", "content": "static threshold duct-tape, no state"},
        ]
    )
    parsed = extract_json(out)
    assert parsed is not None
    assert 0.0 <= float(parsed["deficit_score"]) <= 1.0
    assert isinstance(parsed["red_flags"], list)


def test_extract_json_from_fenced_and_noisy():
    assert extract_json('```json\n{"a": 1}\n```') == {"a": 1}
    assert extract_json('blah blah {"x": [1,2], "y": "z"} trailing') == {"x": [1, 2], "y": "z"}
    assert extract_json("not json at all") is None
    assert extract_json(json.dumps({"k": "v"})) == {"k": "v"}
