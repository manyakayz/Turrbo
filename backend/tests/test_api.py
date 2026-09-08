"""
Integration tests for the FastAPI backend. These require the real C-MAPSS
data/ folder and a trained turbofan_rul_v4.keras model on disk (both
gitignored — not present in CI). Run locally with:

    pytest --override-ini="addopts=" backend/tests/test_api.py

or simply:

    pytest -m integration
"""
import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

DATA_AVAILABLE = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "data"))
MODEL_AVAILABLE = os.path.exists(os.path.join(os.path.dirname(__file__), "..", "..", "turbofan_rul_v4.keras"))

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def client():
    if not (DATA_AVAILABLE and MODEL_AVAILABLE):
        pytest.skip("Real data/ and turbofan_rul_v4.keras not present — see MODELS.md")
    from fastapi.testclient import TestClient
    from backend.app.main import app
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    resp = client.get("/api/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["model_loaded"] is True


def test_get_engines_returns_curated_fleet(client):
    resp = client.get("/api/engines")
    assert resp.status_code == 200
    engines = resp.json()
    assert len(engines) > 0
    for e in engines:
        assert e["status"] in ("Healthy", "Moderate", "Warning", "Critical")
        assert e["risk_level"] in ("Low", "Medium", "High", "Severe")
        assert e["maintenance_category"] in (
            "Healthy", "Inspection Recommended", "Maintenance Required", "Critical"
        )
        assert e["rul_predicted"] >= 0
        assert e["rul_std"] >= 0


def test_get_engine_telemetry_known_unit(client):
    engines = client.get("/api/engines").json()
    unit = engines[0]["original_unit"]
    resp = client.get(f"/api/engines/{unit}/telemetry")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["telemetry"]) > 0
    assert len(body["degradation"]) > 0
    assert body["recommendation"] is not None


def test_get_engine_telemetry_unknown_unit_returns_404(client):
    resp = client.get("/api/engines/999999/telemetry")
    assert resp.status_code == 404


def test_fleet_summary_counts_match_engine_list(client):
    engines = client.get("/api/engines").json()
    summary = client.get("/api/fleet/summary").json()
    assert summary["total_engines"] == len(engines)
    assert sum(summary["by_status"].values()) == len(engines)
