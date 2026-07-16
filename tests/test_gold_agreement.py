"""C: judge-vs-gold agreement (stronger-model proxy as a human-proxy labeler).

The LIVE gold pass needs a served 32B (or a cloud endpoint); these tests exercise
the pure LOGIC deterministically offline — the agreement math, the small held-out
set, and the ADDITIVE surfacing in report.build_report + the metrics ledger — so
the default suite stays offline and green.
"""

import json

import scripts.gold_agreement as ga
from backend.evaluator import report, versioning


def _write(path, obj):
    path.write_text(json.dumps(obj), encoding="utf-8")


# --- held-out set ----------------------------------------------------------
def test_heldout_set_is_small_and_mixes_anchors_and_adversarial():
    items = ga.build_heldout(limit=20)
    assert 1 <= len(items) <= 20
    ids = {i["id"] for i in items}
    assert any(i.startswith("test_") or i.startswith("grid_test_") for i in ids)  # test anchors
    assert any(i.startswith("adv_") for i in ids)                                  # gamed samples
    # labels are well-defined for every held-out item (planted-agreement is meaningful)
    assert all(i.get("label") in ("weak", "strong") for i in items)


# --- agreement math --------------------------------------------------------
def test_compute_agreement_from_synthetic_cache(tmp_path):
    cache = tmp_path / "cache.json"
    result = tmp_path / "gold.json"
    # 4 items: 3 where judge==gold, 1 disagreement; planted labels known.
    _write(cache, {
        "tau": 0.3, "champion_version": "champion-0",
        "items": {
            "a": {"label": "weak",   "judge": {"model": "8b", "using_mock": False, "predicted": "weak",   "deficit": 0.8, "scalar": 0.8, "max_penalty": 0.8},
                                     "gold":  {"model": "32b", "using_mock": False, "predicted": "weak",   "deficit": 0.9, "scalar": 0.9, "max_penalty": 0.9}},
            "b": {"label": "strong", "judge": {"model": "8b", "using_mock": False, "predicted": "strong", "deficit": 0.1, "scalar": 0.1, "max_penalty": 0.1},
                                     "gold":  {"model": "32b", "using_mock": False, "predicted": "strong", "deficit": 0.1, "scalar": 0.1, "max_penalty": 0.1}},
            "c": {"label": "weak",   "judge": {"model": "8b", "using_mock": False, "predicted": "weak",   "deficit": 0.7, "scalar": 0.7, "max_penalty": 0.7},
                                     "gold":  {"model": "32b", "using_mock": False, "predicted": "weak",   "deficit": 0.9, "scalar": 0.9, "max_penalty": 0.9}},
            "d": {"label": "weak",   "judge": {"model": "8b", "using_mock": False, "predicted": "strong", "deficit": 0.1, "scalar": 0.1, "max_penalty": 0.1},
                                     "gold":  {"model": "32b", "using_mock": False, "predicted": "weak",   "deficit": 0.9, "scalar": 0.9, "max_penalty": 0.9}},
        },
    })
    res = ga.compute_agreement(cache, result, tau=0.3)
    assert res is not None
    jvg = res["judge_vs_gold"]
    assert jvg["n"] == 4
    assert jvg["accuracy"] == 0.75                       # 3/4 judge==gold
    assert jvg["confusion"] == {"weak_weak": 2, "weak_strong": 0, "strong_weak": 1, "strong_strong": 1}
    assert -1.0 <= jvg["cohen_kappa"] <= 1.0
    assert res["judge_model"] == "8b" and res["gold_model"] == "32b"
    assert "proxy" in res["caveat"].lower() and "human" in res["caveat"].lower()
    # result was persisted for report.build_report to read
    assert result.exists()


def test_compute_agreement_returns_none_when_incomplete(tmp_path):
    cache = tmp_path / "cache.json"
    _write(cache, {"tau": 0.3, "champion_version": "champion-0",
                   "items": {"a": {"label": "weak", "judge": {"model": "8b", "using_mock": False, "predicted": "weak"}}}})
    assert ga.compute_agreement(cache, tmp_path / "r.json", tau=0.3) is None  # gold pass missing


# --- additive surfacing in report.build_report -----------------------------
def test_report_surfaces_judge_vs_gold_when_present(tmp_path, monkeypatch):
    gold = tmp_path / "gold_agreement.json"
    _write(gold, {
        "champion_version": versioning.get_champion_version(),
        "n": 18, "tau": 0.3, "judge_model": "llama-3.1-8b", "gold_model": "qwen2.5-32b",
        "judge_vs_gold": {"accuracy": 0.88, "cohen_kappa": 0.72, "confusion": {}, "n": 18},
        "judge_vs_planted": {"accuracy": 0.9, "cohen_kappa": 0.8},
        "gold_vs_planted": {"accuracy": 0.95, "cohen_kappa": 0.9},
        "caveat": "gold model is a STRONGER-MODEL proxy, NOT a human labeler.",
    })
    monkeypatch.setattr(report, "GOLD_AGREEMENT_PATH", gold)
    rep = report.build_report(include_agreement=False)
    assert rep["judge_vs_gold"]["gold_model"] == "qwen2.5-32b"
    assert rep["judge_vs_gold"]["judge_vs_gold"]["cohen_kappa"] == 0.72
    assert "human" in rep["judge_vs_gold"]["caveat"].lower()


def test_report_judge_vs_gold_absent_is_empty(tmp_path, monkeypatch):
    monkeypatch.setattr(report, "GOLD_AGREEMENT_PATH", tmp_path / "does_not_exist.json")
    rep = report.build_report(include_agreement=False)
    assert rep["judge_vs_gold"] == {}


def test_report_ignores_stale_cross_epoch_gold(tmp_path, monkeypatch):
    gold = tmp_path / "gold_agreement.json"
    _write(gold, {"champion_version": "champion-FROM-ANOTHER-EPOCH",
                  "judge_vs_gold": {"accuracy": 1.0, "cohen_kappa": 1.0}})
    monkeypatch.setattr(report, "GOLD_AGREEMENT_PATH", gold)
    assert report.build_report(include_agreement=False)["judge_vs_gold"] == {}  # champion mismatch -> ignored


# --- metrics ledger carries the judge-vs-gold fields (additive) ------------
def test_ledger_snapshot_carries_judge_vs_gold_fields(tmp_path, monkeypatch):
    gold = tmp_path / "gold_agreement.json"
    _write(gold, {
        "champion_version": versioning.get_champion_version(),
        "judge_model": "llama-3.1-8b", "gold_model": "qwen2.5-32b",
        "judge_vs_gold": {"accuracy": 0.88, "cohen_kappa": 0.72},
    })
    monkeypatch.setattr(report, "GOLD_AGREEMENT_PATH", gold)
    for key in ("judge_vs_gold_kappa", "judge_vs_gold_accuracy", "gold_model"):
        assert key in report._LEDGER_KEYS
    row = report._snapshot_from_report(report.build_report(include_agreement=False))
    assert row["judge_vs_gold_kappa"] == 0.72
    assert row["judge_vs_gold_accuracy"] == 0.88
    assert row["gold_model"] == "qwen2.5-32b"
