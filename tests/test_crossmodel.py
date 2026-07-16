"""p1-crossmodel: route one panel seat to a DIFFERENT model family.

A same-family judge panel shares self-preference bias + blind spots. The panel's
cross-model seam is now wired (config/env) so one seat can be a different family.
Offline the deterministic mock ignores the model id, so the verdict stays
reproducible; these tests prove the ROUTING (which model each seat uses) and that
offline determinism is preserved.
"""

import json

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import panel, versioning
from backend.inference import lemonade_client as lc
from backend.inference.lemonade_client import LemonadeClient

_OTHER_FAMILY = "qwen2.5-7b-instruct"  # a deliberately different family from the default


def _weak_candidate():
    a = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))[0]
    return anchor_ds.anchor_candidate_text(a), a.get("domain")


def test_resolve_panel_pins_last_seat_to_cross_model():
    base = panel.resolve_panel()
    assert all(len(p) == 2 for p in base)  # same-family by default
    crossed = panel.resolve_panel(cross_model=_OTHER_FAMILY)
    assert len(crossed[-1]) == 3 and crossed[-1][2] == _OTHER_FAMILY
    assert all(len(p) == 2 for p in crossed[:-1])  # only the last seat is cross-family


def test_resolve_panel_reads_env(monkeypatch):
    monkeypatch.setenv(panel.ENV_CROSS_MODEL, _OTHER_FAMILY)
    crossed = panel.resolve_panel()
    assert crossed[-1][2] == _OTHER_FAMILY
    monkeypatch.delenv(panel.ENV_CROSS_MODEL, raising=False)
    assert all(len(p) == 2 for p in panel.resolve_panel())


def test_cross_model_is_offline_deterministic_and_routes_one_seat():
    cand, domain = _weak_candidate()
    champ = versioning.get_champion_rubric_text()
    base = panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0)
    cross = panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, cross_model=_OTHER_FAMILY)
    # Offline the mock ignores model id => identical verdict (determinism preserved).
    assert cross.deficit_median == base.deficit_median
    assert cross.verdict == base.verdict
    # ...but exactly one seat is routed to the different family.
    assert [pp["model"] for pp in base.per_persona] == [None, None, None]
    assert [pp["model"] for pp in cross.per_persona].count(_OTHER_FAMILY) == 1


class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def test_cross_model_is_threaded_to_the_live_client(monkeypatch):
    seen_models: list[str] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        seen_models.append(json.get("model"))
        return _FakeResp(json_module.dumps({"deficit_loose": 0.5, "criterion_penalties": {}, "red_flags": []}))

    json_module = json
    monkeypatch.setattr(lc.httpx, "post", fake_post)
    monkeypatch.setattr(lc.httpx, "get", lambda url, headers=None, timeout=None: type("R", (), {"status_code": 200})())

    client = LemonadeClient(base_url="http://live.invalid/v1", force_mock=False)
    cand, domain = _weak_candidate()
    champ = versioning.get_champion_rubric_text()
    panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, client=client, seed=1, cross_model=_OTHER_FAMILY)

    # The different-family model was actually used on exactly one seat; the others
    # used the client's primary model (not the cross-model).
    assert seen_models.count(_OTHER_FAMILY) == 1
    assert any(m != _OTHER_FAMILY for m in seen_models)
