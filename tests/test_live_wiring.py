"""p1_live: live-model wiring verified OFFLINE + real-model tests behind @live.

Everything here runs with NO GPU and NO network:

* the record/replay **cassette** layer is unit-tested directly;
* the **live HTTP path** is exercised through a fake ``httpx`` transport so the
  structured-output contract, the ``seed``/``response_format`` threading and the
  strict-vs-loose (hack-ratio) differentiation are all checked without a model;
* a recorded transcript is **replayed** with the network hard-disabled to prove
  reproducibility;
* ``rqgm`` provenance + the exact API surface the adapter depends on are pinned;
* the challenger version id is asserted reproducible.

The genuinely-needs-a-model tests carry ``@pytest.mark.live`` and are skipped by
default (see ``tests/conftest.py``); run them with a server up::

    RQGM_RUN_LIVE=1 LEMONADE_BASE_URL=http://localhost:8000/v1 \
        uv run pytest -m live
"""

import json
import os

import pytest

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import evolve, panel, report, rqgm_adapter, versioning
from backend.evaluator.judge import evaluate_architecture, judge_response_format, score_candidate
from backend.inference import lemonade_client as lc
from backend.inference.cassette import (
    AUTO,
    OFF,
    RECORD,
    REPLAY,
    Cassette,
    request_key,
)
from backend.inference.lemonade_client import DEFAULT_MODEL, LemonadeClient

_DOMAIN = "smart_manufacturing"
_LIVE_BASE = "http://live.invalid/v1"


# ---------------------------------------------------------------------------
# Fake transport helpers (simulate a live OpenAI-compatible server, offline)
# ---------------------------------------------------------------------------
class _FakeResp:
    def __init__(self, content: str) -> None:
        self._content = content

    def raise_for_status(self) -> None:  # noqa: D401 - httpx.Response shim
        return None

    def json(self) -> dict:
        return {"choices": [{"message": {"content": self._content}}]}


def _install_fake_transport(monkeypatch, responder):
    """Patch httpx so the client talks to ``responder(payload) -> completion``.

    Returns the list of POST payloads seen (for assertions).
    """
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append(json)
        return _FakeResp(responder(json))

    class _Ok:
        status_code = 200

    def fake_get(url, headers=None, timeout=None):
        return _Ok()

    monkeypatch.setattr(lc.httpx, "post", fake_post)
    monkeypatch.setattr(lc.httpx, "get", fake_get)
    return calls


def _disable_network(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("network used where it must not be (replay/offline)")

    monkeypatch.setattr(lc.httpx, "post", boom)
    monkeypatch.setattr(lc.httpx, "get", boom)


def _is_strict(payload: dict) -> bool:
    return any("SCORING_MODE: strict" in m.get("content", "") for m in payload.get("messages", []))


def _strict_loose_responder(payload: dict) -> str:
    """High quality under loose scoring, low quality under strict (poison pills).

    The loose reply deliberately omits ``deficit_strict`` so ``score_candidate``
    makes the second (strict) call — exactly the live two-call path.
    """
    if _is_strict(payload):
        return json.dumps({
            "deficit_strict": 0.75,
            "criterion_penalties": {"reward_hacking_resistance": 0.75},
            "red_flags": [{"criterion": "reward_hacking_resistance", "severity": "high", "detail": "pill"}],
            "reasoning": "strict: poison pill applied",
        })
    return json.dumps({
        "deficit_loose": 0.1,
        "criterion_penalties": {"reward_hacking_resistance": 0.1},
        "red_flags": [{"criterion": "reward_hacking_resistance", "severity": "low", "detail": "minor"}],
        "reasoning": "loose: mostly fine",
    })


# ---------------------------------------------------------------------------
# 1) Cassette record/replay layer (pure unit, offline, no model)
# ---------------------------------------------------------------------------
def test_cassette_round_trip_and_stable_key(tmp_path):
    cas = Cassette(tmp_path, REPLAY)
    payload = {"model": "m", "messages": [{"role": "user", "content": "hi"}], "temperature": 0.2}
    # key is order-independent (canonicalised) + stable.
    reordered = {"temperature": 0.2, "messages": [{"role": "user", "content": "hi"}], "model": "m"}
    assert request_key(payload) == request_key(reordered)
    assert cas.get(payload) is None  # miss before record

    key = cas.put(payload, "COMPLETION-A")
    assert key == request_key(payload)
    assert cas.get(payload) == "COMPLETION-A"
    # a different request does not collide.
    assert cas.get({**payload, "temperature": 0.9}) is None


def test_cassette_mode_gating(tmp_path):
    assert Cassette(tmp_path, REPLAY).can_replay is True
    assert Cassette(tmp_path, REPLAY).can_record is False
    assert Cassette(tmp_path, RECORD).can_record is True
    assert Cassette(tmp_path, RECORD).can_replay is False
    assert Cassette(tmp_path, AUTO).can_replay and Cassette(tmp_path, AUTO).can_record
    with pytest.raises(ValueError):
        Cassette(tmp_path, "bogus")


def test_cassette_from_env(tmp_path, monkeypatch):
    monkeypatch.delenv("LEMONADE_CASSETTE_DIR", raising=False)
    assert Cassette.from_env() is None  # no dir -> disabled
    monkeypatch.setenv("LEMONADE_CASSETTE_DIR", str(tmp_path))
    monkeypatch.setenv("LEMONADE_CASSETTE_MODE", OFF)
    assert Cassette.from_env() is None  # explicit off
    monkeypatch.setenv("LEMONADE_CASSETTE_MODE", REPLAY)
    cas = Cassette.from_env()
    assert cas is not None and cas.mode == REPLAY and cas.directory == tmp_path


# ---------------------------------------------------------------------------
# 2) Live HTTP path + using_mock semantics (fake transport, offline)
# ---------------------------------------------------------------------------
def test_live_http_path_reports_not_mock_and_threads_seed_and_schema(monkeypatch):
    calls = _install_fake_transport(monkeypatch, lambda p: "LIVE-OK")
    client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False)
    assert client.using_mock is False  # a reachable server is not the mock
    out = client.chat("hello", seed=123, response_format=judge_response_format())
    assert out == "LIVE-OK"
    assert calls[-1]["seed"] == 123
    assert calls[-1]["response_format"]["type"] == "json_schema"


def test_force_mock_never_touches_network(monkeypatch):
    _disable_network(monkeypatch)  # any httpx use would raise
    client = LemonadeClient(force_mock=True)
    assert client.using_mock is True
    assert client.chat("hello").startswith("[MOCK")


def test_unreachable_server_falls_back_to_mock_without_hanging(monkeypatch):
    # Simulate an unreachable endpoint: httpx raises on connect.
    def boom_get(*a, **k):
        raise lc.httpx.ConnectError("refused")

    monkeypatch.setattr(lc.httpx, "get", boom_get)
    client = LemonadeClient(base_url="http://127.0.0.1:59999/v1", force_mock=False)
    assert client.using_mock is True  # probe failed -> mock
    assert isinstance(client.chat("hello"), str)


# ---------------------------------------------------------------------------
# 3) Structured judge output on the live path (fake transport, offline)
# ---------------------------------------------------------------------------
def test_live_judge_uses_structured_criterion_penalties(monkeypatch):
    calls = _install_fake_transport(monkeypatch, _strict_loose_responder)
    client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False)
    ev = evaluate_architecture(
        "duct-tape threshold agent", domain_id=_DOMAIN, client=client,
        inject_memory=False, seed=7,
    )
    assert ev.used_mock is False
    # The structured contract was sent...
    assert calls[0]["seed"] == 7
    assert calls[0]["response_format"]["type"] == "json_schema"
    # ...and the model's EXPLICIT penalty (0.1) is used, NOT the flag-severity
    # approximation (severity "low" -> 0.2), so sep::<criterion> stays faithful.
    assert ev.criterion_penalties == {"reward_hacking_resistance": 0.1}


def test_live_strict_vs_loose_differentiates_hack_ratio(monkeypatch):
    _install_fake_transport(monkeypatch, _strict_loose_responder)
    client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False)
    res = score_candidate("gamed candidate", versioning.get_champion_rubric_text(),
                          domain_id=_DOMAIN, epoch=0, client=client, seed=3)
    # The live path applies poison pills on the strict call, so hack_ratio is NOT
    # trivially ~1 (this is what keeps strict-vs-loose meaningful on real models).
    assert res["deficit_strict"] > res["deficit_loose"]
    assert res["hack_ratio"] is not None and res["hack_ratio"] < 1.0


def test_malformed_live_output_falls_back_to_flag_penalties(monkeypatch):
    # Model ignores the schema and returns criterion_penalties as a non-dict:
    def bad(payload):
        return json.dumps({
            "deficit_loose": 0.4,
            "criterion_penalties": "not-a-dict",
            "red_flags": [{"criterion": "diagnostic_resilience", "severity": "high", "detail": "x"}],
        })

    _install_fake_transport(monkeypatch, bad)
    client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False)
    ev = evaluate_architecture("x", domain_id=_DOMAIN, client=client, inject_memory=False)
    assert ev.used_mock is False
    # Graceful fallback: penalties approximated from red-flag severities (high -> 0.75).
    assert ev.criterion_penalties.get("diagnostic_resilience") == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# 4) Record -> replay reproducibility (fake transport, then network disabled)
# ---------------------------------------------------------------------------
def test_record_then_replay_is_reproducible_offline(tmp_path, monkeypatch):
    _install_fake_transport(monkeypatch, _strict_loose_responder)
    champ = versioning.get_champion_rubric_text()

    rec_client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False, cassette=Cassette(tmp_path, RECORD))
    res_rec = score_candidate("gamed candidate", champ, domain_id=_DOMAIN, epoch=0, client=rec_client, seed=3)
    assert rec_client.using_mock is False
    assert res_rec["deficit_strict"] > res_rec["deficit_loose"]
    assert len(list(tmp_path.glob("*.json"))) >= 2  # loose + strict recorded

    # Now REPLAY with the network hard-disabled: identical result, no model.
    _disable_network(monkeypatch)
    rep_client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False, cassette=Cassette(tmp_path, REPLAY))
    res_rep = score_candidate("gamed candidate", champ, domain_id=_DOMAIN, epoch=0, client=rep_client, seed=3)
    assert rep_client.using_mock is False  # replay serves recorded (not the mock)
    assert res_rep == res_rec


def test_replay_miss_falls_back_to_mock(tmp_path, monkeypatch):
    _disable_network(monkeypatch)
    client = LemonadeClient(base_url=_LIVE_BASE, force_mock=False, cassette=Cassette(tmp_path, REPLAY))
    # Empty cassette -> miss -> deterministic mock (CI never stalls / needs net).
    out = client.chat("anything", seed=1)
    assert isinstance(out, str) and out


# ---------------------------------------------------------------------------
# 5) strict-vs-loose on the MOCK path (item 6: not trivially ~1)
# ---------------------------------------------------------------------------
def test_mock_path_strict_vs_loose_differentiates():
    champ = versioning.get_champion_rubric_text()
    client = LemonadeClient(force_mock=True)
    weak = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))
    results = [
        score_candidate(anchor_ds.anchor_candidate_text(a), champ, domain_id=a.get("domain"), epoch=0, client=client)
        for a in weak
    ]
    # champion-0 has a poison-pill blind spot on the gamed anchors, so at least one
    # weak anchor scores strict > loose and drives hack_ratio below 1.
    assert any(r["deficit_strict"] > r["deficit_loose"] for r in results)
    assert any(r["hack_ratio"] is not None and r["hack_ratio"] < 1.0 for r in results)


# ---------------------------------------------------------------------------
# 6) Panel bias controls (AB/BA swap + seed) stay deterministic offline
# ---------------------------------------------------------------------------
def test_panel_position_swap_is_noop_on_deterministic_mock():
    champ = versioning.get_champion_rubric_text()
    a = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))[0]
    cand = anchor_ds.anchor_candidate_text(a)
    base = panel.evaluate_panel(cand, champ, domain_id=a.get("domain"), epoch=0)
    swapped = panel.evaluate_panel(cand, champ, domain_id=a.get("domain"), epoch=0, position_swap=True)
    # The offline mock is criterion-order invariant, so AB/BA averaging cannot
    # change the verdict -> offline behaviour stays deterministic.
    assert swapped.deficit_median == base.deficit_median
    assert base.per_persona[0]["orientations"] == 1
    assert swapped.per_persona[0]["orientations"] == 2


def test_panel_seed_is_reproducible_and_keeps_diversity():
    champ = versioning.get_champion_rubric_text()
    a = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))[0]
    cand = anchor_ds.anchor_candidate_text(a)
    p1 = panel.evaluate_panel(cand, champ, domain_id=a.get("domain"), epoch=0, seed=42)
    p2 = panel.evaluate_panel(cand, champ, domain_id=a.get("domain"), epoch=0, seed=42)
    assert p1.to_dict() == p2.to_dict()  # reproducible
    # A fixed seed does NOT collapse the panel to one opinion (diversity kept).
    assert len({pp["deficit"] for pp in p1.per_persona}) >= 2


# ---------------------------------------------------------------------------
# 7) rqgm provenance + API surface (NON-live)
# ---------------------------------------------------------------------------
def test_rqgm_backend_is_the_real_package():
    assert rqgm_adapter.RQGM_BACKEND == "rqgm"
    assert rqgm_adapter._RQGM_AVAILABLE is True


def test_rqgm_api_surface_matches_assumptions():
    from rqgm import EpochConfig, EpochManager, TransitionReason

    # The exact enum member the adapter compares against.
    assert hasattr(TransitionReason, "EXPLOITATION_DETECTED")

    cfg = EpochConfig(epoch_size=3, min_improvement_threshold=0.0, exploitation_hack_ratio_threshold=0.6)
    mgr = EpochManager(cfg, initial_tolerances=[0.0, 0.001, 0.01, 0.025, 0.05, 0.1])
    assert hasattr(mgr, "record_iteration_result")
    assert hasattr(mgr, "evaluate_epoch_boundary")

    # Gamed samples (high loose quality, low strict quality) => exploitation.
    strict_q, loose_q = [0.2, 0.15, 0.25], [0.9, 0.95, 0.9]
    for i in range(3):
        mgr.record_iteration_result(i, 1.0, strict_q[i], loose_q[i])
    transition = mgr.evaluate_epoch_boundary(2)
    assert transition.reason == TransitionReason.EXPLOITATION_DETECTED
    assert isinstance(list(transition.new_tolerances), list)
    assert len(transition.new_tolerances) < 6  # tightened (dropped a level)
    assert isinstance(transition.trigger_adversarial_injection, bool)


def test_report_records_judge_model_and_git_sha():
    rep = report.build_report(include_agreement=False)
    prov = rep["provenance"]
    assert prov["rqgm_backend"] == "rqgm"
    assert prov["using_mock"] is True  # offline default
    assert isinstance(prov["judge_model"], str) and prov["judge_model"]
    assert prov["git_sha"] is None or (isinstance(prov["git_sha"], str) and len(prov["git_sha"]) >= 7)
    # Additive only: existing top-level + sub-dict shapes are untouched.
    assert rep["rqgm_backend"] == "rqgm"
    assert set(rep["separation"].keys()) == {"val", "test"}


# ---------------------------------------------------------------------------
# 8) Reproducible challenger version id (no wall-clock nondeterminism)
# ---------------------------------------------------------------------------
def test_challenger_version_id_is_reproducible():
    v1 = evolve.propose_challenger().version
    v2 = evolve.propose_challenger().version
    assert v1 == v2  # content-derived, not time.time()
    assert v1.startswith("challenger-e0-")
    # An injected seed changes the id deterministically.
    a = evolve.propose_challenger(seed="alpha").version
    b = evolve.propose_challenger(seed="beta").version
    assert a != b
    assert a == evolve.propose_challenger(seed="alpha").version


# ---------------------------------------------------------------------------
# 9) REAL-model tests — skipped by default (need a GPU + server)
# ---------------------------------------------------------------------------
@pytest.mark.live
def test_live_evaluator_returns_structured_verdict():
    base = os.getenv("LEMONADE_BASE_URL")
    client = LemonadeClient(base_url=base, force_mock=False)
    if not client.is_live():
        pytest.skip(f"no live model reachable at {base or DEFAULT_MODEL}")
    ev = evaluate_architecture(
        "A LangGraph predictive-maintenance agent with typed state, a physical "
        "root-cause node, and a deterministic HITL safety gate.",
        domain_id=_DOMAIN, client=client, inject_memory=False, seed=7,
    )
    assert ev.used_mock is False
    assert 0.0 <= ev.deficit_score <= 1.0
    assert isinstance(ev.criterion_penalties, dict)


@pytest.mark.live
def test_live_panel_stays_diverse():
    base = os.getenv("LEMONADE_BASE_URL")
    client = LemonadeClient(base_url=base, force_mock=False)
    if not client.is_live():
        pytest.skip(f"no live model reachable at {base or DEFAULT_MODEL}")
    champ = versioning.get_champion_rubric_text()
    a = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))[0]
    pe = panel.evaluate_panel(
        anchor_ds.anchor_candidate_text(a), champ, domain_id=_DOMAIN, epoch=0,
        client=client, seed=11, position_swap=True,
    )
    assert pe.n_judges == 3
    assert len(pe.per_persona) == 3
