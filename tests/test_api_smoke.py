"""Backend smoke test: import the app and exercise the contract via TestClient.

Runs entirely on the inference MOCK (no live Lemonade / Qdrant server needed).
"""

import pytest
from fastapi.testclient import TestClient

from backend.app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


def test_health_and_root(client):
    h = client.get("/health").json()
    assert h["status"] == "ok"
    assert h["inference"]["using_mock"] is True
    assert client.get("/").json()["name"] == "AgentForge"
    # Phase 4: /health exposes a compact evaluator summary.
    ev = h["evaluator"]
    assert "val_separation" in ev and "test_separation" in ev
    assert "hack_ratio" in ev and "exploitation_detected" in ev
    assert "val_judge_accuracy" in ev
    assert ev["rqgm_backend"] in {"rqgm", "local-fallback"}


def test_admin_report_endpoint(client):
    r = client.get("/api/admin/report").json()
    assert set(r["separation"].keys()) == {"val", "test"}
    assert r["separation"]["val"]["separation"] >= 0
    assert "mean_hack_ratio" in r["hack_ratio"]
    assert "accuracy" in r["judge_agreement"]["val"]
    assert "cohen_kappa" in r["judge_agreement"]["test"]
    assert r["data_splits"]["test"]["total"] == 9
    assert r["rqgm_backend"] in {"rqgm", "local-fallback"}


def test_tiers_have_class_key(client):
    tiers = client.get("/api/tiers").json()["tiers"]
    assert len(tiers) >= 5
    for t in tiers:
        assert "class" in t  # contract key (not "tier_class")
        assert t["class"] in {"ryzen_ai", "radeon", "radeon_pro", "instinct"}
        assert t["memory_gb"] > 0
        assert t["bandwidth_tbs"] > 0


def test_models_catalog(client):
    models = client.get("/api/models").json()["models"]
    assert len(models) >= 5
    m = models[0]
    for key in ("id", "name", "params_b", "n_layers", "n_kv_heads", "head_dim", "hidden", "context_len", "dtype_default"):
        assert key in m


def _new_session(client) -> str:
    return client.post("/api/session").json()["session_id"]


def test_domain_routing(client):
    sid = _new_session(client)
    r = client.post(
        f"/api/session/{sid}/domain",
        json={"domain": "smart_manufacturing", "description": "predictive maintenance for bearing vibration anomaly"},
    ).json()
    assert r["matched_templates"]
    assert r["recommended_template_id"] == "predictive_maintenance"


def test_diagnose_feasible_and_infeasible(client):
    sid = _new_session(client)
    ok = client.post(
        f"/api/session/{sid}/diagnose",
        json={
            "current_hardware": {"tier_id": "rx_7900_xtx"},
            "requirements": {"model_id": "llama-3.1-8b", "seq_len": 8192, "concurrency": 2, "dtype": "int4"},
        },
    ).json()
    assert ok["feasible"] is True
    assert ok["report"]["vram_total_gb"] > 0
    bd = ok["report"]["vram_breakdown"]
    assert set(bd.keys()) == {"weights", "kv_cache", "activations", "overhead"}

    bad = client.post(
        f"/api/session/{sid}/diagnose",
        json={
            "current_hardware": {"custom": {"memory_gb": 8, "bandwidth_tbs": 0.2}},
            "requirements": {"model_id": "llama-3.1-70b", "seq_len": 8192, "concurrency": 1, "dtype": "fp16"},
        },
    ).json()
    assert bad["feasible"] is False
    assert any(g["constraint"] == "vram_gb" for g in bad["gaps"])


def test_simulate_all_tiers(client):
    sid = _new_session(client)
    r = client.post(
        f"/api/session/{sid}/simulate",
        json={"model_id": "llama-3.1-8b", "seq_len": 8192, "population": 16, "dtype": "int4", "prefix_ratio": 0.9},
    ).json()
    per_tier = r["per_tier"]
    assert len(per_tier) >= 5
    mi = next(t for t in per_tier if t["tier_id"] == "mi300x")
    assert mi["feasible"] is True
    assert mi["max_population"] > 100
    assert mi["kv_savings_from_prefix_pct"] > 0


def test_evaluate_returns_deficit(client):
    sid = _new_session(client)
    r = client.post(
        f"/api/session/{sid}/evaluate",
        json={"architecture": "single static threshold shutdown, no state schema", "domain": "smart_manufacturing"},
    ).json()
    assert 0.0 <= r["deficit_score"] <= 1.0
    assert isinstance(r["red_flags"], list)
    assert isinstance(r["epoch_id"], int)


def test_export_returns_tco_and_files(client):
    sid = _new_session(client)
    client.post(
        f"/api/session/{sid}/domain",
        json={"domain": "smart_manufacturing", "description": "predictive maintenance"},
    )
    r = client.post(
        f"/api/session/{sid}/export",
        json={"target_tier_id": "mi300x", "model_id": "llama-3.1-8b", "template_id": "predictive_maintenance"},
    ).json()
    assert "AMD TCO" in r["tco_markdown"]
    assert "docker-compose.yml" in r["deploy_files"]
    assert "rocm/vllm-dev" in r["deploy_files"]["docker-compose.yml"]


def test_feedback_stored(client):
    sid = _new_session(client)
    r = client.post(
        f"/api/session/{sid}/feedback",
        json={"rating": 2, "correct": False, "notes": "threshold false-tripped on correlated noise"},
    ).json()
    assert r["ok"] is True
    assert r["stored_as"] == "ground_truth_anchor"


def test_epoch_propose_then_approve_advances_epoch(client):
    before = client.get("/health").json()["epoch_id"]
    prop = client.post("/api/admin/epoch/propose").json()
    assert prop["challenger_id"].startswith("challenger-")
    assert "metrics" in prop and "rubric_diff" in prop
    assert "frontier" in prop  # Phase 2: propose returns the Pareto frontier
    appr = client.post("/api/admin/epoch/approve", json={"approve": True}).json()
    # Two-stage gate: code gate must pass, and it is surfaced in the response.
    assert appr["gate"]["passed"] is True
    assert appr["gate"]["challenger_separation"] >= appr["gate"]["champion_separation"]
    assert appr["hitl"]["consulted"] is True
    assert appr["applied"] is True
    assert appr["epoch_id"] == before + 1


def test_epoch_approve_rejects_worse_challenger_before_hitl(client):
    """Code gate blocks a strictly-worse challenger even if a human approves."""
    from backend.evaluator import versioning

    versioning.reset()
    champion = versioning.get_champion_rubric_text()
    worse = champion.replace(
        '<criterion id="safety_autonomy" weight="0.15">',
        '<criterion id="_removed" weight="0.15">',
    )
    versioning.register_challenger(version="challenger-worse-api", rubric_text=worse, parent_version="champion-0")
    # Seed the admin module's pending pointer directly to the worse challenger.
    from backend.app.api import admin as admin_mod

    admin_mod._pending_challenger = "challenger-worse-api"
    appr = client.post("/api/admin/epoch/approve", json={"approve": True}).json()
    assert appr["applied"] is False
    assert appr["gate"]["passed"] is False
    assert appr["hitl"]["consulted"] is False


def test_orchestrate_hitl_interrupt_and_resume(client):
    sid = _new_session(client)
    client.post(
        f"/api/session/{sid}/domain",
        json={"domain": "smart_manufacturing", "description": "predictive maintenance"},
    )
    started = client.post(
        f"/api/session/{sid}/orchestrate",
        json={"need": "predictive maintenance", "model_id": "llama-3.1-8b", "tier_id": "rx_7900_xtx", "seq_len": 8192, "concurrency": 2, "dtype": "int4"},
    ).json()
    assert started["awaiting_hitl"] is True
    assert "hitl" in started["next"]
    resumed = client.post(
        f"/api/session/{sid}/orchestrate/resume",
        json={"approved": True, "notes": "looks good"},
    ).json()
    assert resumed["awaiting_hitl"] is False
    assert resumed["state"]["approved"] is True


def test_orchestrate_hard_reject_on_infeasible(client):
    sid = _new_session(client)
    started = client.post(
        f"/api/session/{sid}/orchestrate",
        json={"need": "huge model", "model_id": "llama-3.1-70b", "tier_id": "rx_7900_xtx", "seq_len": 8192, "concurrency": 1, "dtype": "fp16"},
    ).json()
    # Physics gate: infeasible -> no HITL, hard rejection recorded.
    assert started["awaiting_hitl"] is False
    assert started["state"]["feasible"] is False
    assert "HARD REJECT" in (started["state"].get("gate_rejection") or "")
