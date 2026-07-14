"""Tests for domain-pack auto-discovery and the need->template router."""

from backend.domains.registry import get_domain, list_domains
from backend.graph.router import best_template, route_need


def test_smart_manufacturing_is_discovered():
    ids = [d.id for d in list_domains()]
    assert "smart_manufacturing" in ids


def test_domain_pack_contents():
    pack = get_domain("smart_manufacturing")
    assert pack is not None
    assert len(pack.poison_pills()) >= 3
    template_ids = {t.id for t in pack.workflow_templates()}
    assert {"predictive_maintenance", "visual_qc", "process_optimization"} <= template_ids
    assert "criterion" in pack.rubric_fragment()


def test_router_maps_needs_to_templates():
    assert best_template("predict bearing failure from vibration telemetry", "smart_manufacturing") == "predictive_maintenance"
    assert best_template("camera inspection of surface scratches and defects", "smart_manufacturing") == "visual_qc"
    assert best_template("optimize setpoints for maximum yield throughput", "smart_manufacturing") == "process_optimization"


def test_router_returns_ranked_scores():
    matches = route_need("defect image inspection quality", "smart_manufacturing")
    assert matches
    assert matches[0].score >= matches[-1].score  # sorted descending
