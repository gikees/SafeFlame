"""Tests for FastAPI dashboard endpoints."""

from unittest.mock import MagicMock, patch

import pytest

import config


@pytest.fixture
def client():
    """FastAPI TestClient for the dashboard."""
    from fastapi.testclient import TestClient
    from dashboard.server import app, set_shared_state, _shared
    # Reset shared state
    _shared["state_machine"] = None
    _shared["alert_manager"] = None
    _shared["fps"] = 0.0
    _shared["inference_ms"] = 0.0
    return TestClient(app)


class TestRootEndpoint:
    def test_root_returns_html(self, client):
        resp = client.get("/")
        assert resp.status_code == 200
        assert "SafeFlame" in resp.text


class TestStatusEndpoint:
    def test_status_no_state_machine(self, client):
        resp = client.get("/api/status")
        assert resp.status_code == 200
        data = resp.json()
        assert data["person_present"] is False
        assert data["zones"] == {}

    def test_status_with_state_machine(self, client):
        from dashboard.server import set_shared_state
        mock_sm = MagicMock()
        mock_sm.get_status.return_value = {
            "person_present": True,
            "person_absent_seconds": None,
            "zones": {"B1": {"state": "active_attended", "unattended_seconds": None}},
        }
        set_shared_state("state_machine", mock_sm)
        set_shared_state("fps", 30.0)
        set_shared_state("inference_ms", 12.5)
        resp = client.get("/api/status")
        data = resp.json()
        assert data["person_present"] is True
        assert data["fps"] == 30.0
        assert "B1" in data["zones"]


class TestAlertsEndpoint:
    def test_alerts_no_manager(self, client):
        resp = client.get("/api/alerts")
        assert resp.status_code == 200
        assert resp.json() == []

    def test_alerts_with_manager(self, client):
        from dashboard.server import set_shared_state
        mock_am = MagicMock()
        mock_am.get_recent_alerts.return_value = [
            {"type": "smoke", "severity": "critical", "message": "Smoke!", "timestamp": 1000.0}
        ]
        set_shared_state("alert_manager", mock_am)
        resp = client.get("/api/alerts")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["type"] == "smoke"


class TestZonesEndpoint:
    def test_get_zones_empty(self, client):
        config.BURNER_ZONES = []
        resp = client.get("/api/zones")
        assert resp.json() == []

    def test_post_zones(self, client):
        zones = [{"name": "B1", "x": 100, "y": 100, "w": 200, "h": 200}]
        resp = client.post("/api/zones", json=zones)
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert len(data["zones"]) == 1
        # Config should be updated
        assert len(config.BURNER_ZONES) == 1

    def test_get_zones_after_post(self, client):
        zones = [{"name": "FL", "x": 10, "y": 20, "w": 100, "h": 100}]
        client.post("/api/zones", json=zones)
        resp = client.get("/api/zones")
        data = resp.json()
        assert len(data) == 1
        assert data[0]["name"] == "FL"


class TestConfigEndpoint:
    def test_update_known_config(self, client):
        resp = client.post("/api/config", json={"key": "YOLO_CONFIDENCE", "value": 0.6})
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"
        assert config.YOLO_CONFIDENCE == 0.6

    def test_update_unknown_config(self, client):
        resp = client.post("/api/config", json={"key": "NONEXISTENT_KEY", "value": 42})
        data = resp.json()
        assert data["status"] == "error"
