"""p2-mcts-debate (optional): shallow-MCTS frontier selection + panel debate.

Both are opt-in. The promotion GATE stays deterministic and debate-free (it uses
score_candidate, never the panel). Offline the debate round is a reproducible
no-op (the mock ignores the injected rebuttal), matching the repo's honesty
boundary for live-only behaviour.
"""

import json

from backend.evaluator import anchors as anchor_ds
from backend.evaluator import evolve, gate, panel, versioning
from backend.evaluator.frontier import FrontierMember, ParetoFrontier
from backend.evaluator.mutation import mutate_rubric_text
from backend.inference import lemonade_client as lc
from backend.inference.lemonade_client import LemonadeClient
from backend.memory.qdrant_store import EvolutionaryMemory

_DOMAIN = "smart_manufacturing"


# --- shallow MCTS (UCT) parent selection -----------------------------------
def test_sample_uct_expands_unvisited_then_exploits():
    fr = ParetoFrontier(top_k=8)
    a = FrontierMember(version="a", rubric_text="", objectives={"x": 1.0}, bbe=0.1)
    b = FrontierMember(version="b", rubric_text="", objectives={"y": 1.0}, bbe=0.1)
    fr.members = [a, b]
    assert fr.sample_uct().version == "a"  # unvisited nodes expanded first (in order)
    for _ in range(10):
        fr.record_child_outcome(a, improved=True)   # a parents improving children
        fr.record_child_outcome(b, improved=False)  # b never does
    assert fr.sample_uct().version == "a"  # UCT exploits the higher success-rate node


def test_gepa_evolve_mcts_strategy_finds_improving_frontier():
    fr = evolve.gepa_evolve(
        versioning.get_champion_rubric_text(), epoch=0, budget=6,
        adversarial_samples=evolve.generate_adversarial_pool(), strategy="mcts",
    )
    best = fr.best()
    assert best is not None and best.version != "champion-0"
    assert best.added_criteria


def test_propose_via_frontier_mcts_still_passes_the_gate():
    prop, _fr = evolve.propose_via_frontier(
        adversarial_samples=evolve.generate_adversarial_pool(), strategy="mcts"
    )
    res = evolve.approve_challenger(
        prop.version, approve=True, memory=EvolutionaryMemory(collection="t_mcts")
    )
    assert res["gate"]["passed"] is True
    assert res["applied"] is True
    assert versioning.get_epoch() == 1


# --- multi-agent debate (opt-in, high-disagreement only) -------------------
def _weak_cand():
    a = anchor_ds.weak(anchor_ds.load_anchors(anchor_ds.VAL))[0]
    return anchor_ds.anchor_candidate_text(a), a.get("domain")


def test_debate_offline_is_a_deterministic_noop_but_flagged():
    cand, domain = _weak_cand()
    champ = versioning.get_champion_rubric_text()
    base = panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0)
    # Force the trigger (threshold 0) — offline the mock ignores the rebuttal, so the
    # verdict/median are unchanged (reproducible), but the round is recorded.
    deb = panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, debate=True, debate_threshold=0.0)
    assert deb.debate_triggered is True and deb.debate_rounds == 1
    assert deb.deficit_median == base.deficit_median
    assert deb.verdict == base.verdict


def test_debate_not_triggered_below_threshold():
    cand, domain = _weak_cand()
    champ = versioning.get_champion_rubric_text()
    deb = panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, debate=True, debate_threshold=0.99)
    assert deb.debate_triggered is False and deb.debate_rounds == 0


class _FakeResp:
    def __init__(self, content):
        self._content = content

    def raise_for_status(self):
        return None

    def json(self):
        return {"choices": [{"message": {"content": self._content}}]}


def test_debate_issues_a_second_round_of_calls_on_the_live_path(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, json=None, headers=None, timeout=None):  # noqa: A002
        calls.append(json)
        return _FakeResp(json_mod.dumps({"deficit_loose": 0.5, "criterion_penalties": {}, "red_flags": []}))

    json_mod = json
    monkeypatch.setattr(lc.httpx, "post", fake_post)
    monkeypatch.setattr(lc.httpx, "get", lambda url, headers=None, timeout=None: type("R", (), {"status_code": 200})())
    client = LemonadeClient(base_url="http://live.invalid/v1", force_mock=False)
    cand, domain = _weak_cand()
    champ = versioning.get_champion_rubric_text()

    calls.clear()
    panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, client=client, seed=1)
    n_no_debate = len(calls)

    calls.clear()
    panel.evaluate_panel(cand, champ, domain_id=domain, epoch=0, client=client, seed=1,
                         debate=True, debate_threshold=0.0)
    assert len(calls) == 2 * n_no_debate  # debate adds exactly one more round


# --- the gate stays deterministic + debate-free ----------------------------
def test_gate_is_deterministic_and_debate_free():
    champ = versioning.get_champion_rubric_text()
    better = mutate_rubric_text(
        champ, [{"id": "reward_hacking_resistance", "text": "Penalize KPI gaming via sensor disabling."}],
        "c-detfree", 1,
    )
    r1 = gate.evaluate_promotion(champ, better, domain_id=_DOMAIN, epoch=0)
    r2 = gate.evaluate_promotion(champ, better, domain_id=_DOMAIN, epoch=0)
    assert r1.to_dict() == r2.to_dict()  # no panel/debate/RNG in the gate
    assert r1.passed is True
