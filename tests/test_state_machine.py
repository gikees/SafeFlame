"""Tests for KitchenStateMachine — escalation, de-escalation, cooldowns, alerts."""

from unittest.mock import patch

import pytest

from state_machine import KitchenStateMachine, BurnerState, Severity, Alert


# ── Enum sanity ──────────────────────────────────────────────────────────────

class TestEnums:
    def test_burner_state_values(self):
        assert BurnerState.OFF.value == "off"
        assert BurnerState.ACTIVE_ATTENDED.value == "active_attended"
        assert BurnerState.ACTIVE_UNATTENDED.value == "active_unattended"
        assert BurnerState.ALERT_INFO.value == "alert_info"
        assert BurnerState.ALERT_WARNING.value == "alert_warning"
        assert BurnerState.ALERT_CRITICAL.value == "alert_critical"

    def test_severity_values(self):
        assert Severity.INFO.value == "info"
        assert Severity.WARNING.value == "warning"
        assert Severity.CRITICAL.value == "critical"

    def test_alert_to_dict(self):
        a = Alert(type="smoke", severity=Severity.CRITICAL, message="Smoke!", timestamp=1000.0, burner_zone="kitchen")
        d = a.to_dict()
        assert d["type"] == "smoke"
        assert d["severity"] == "critical"
        assert d["timestamp"] == 1000.0


# ── Person tracking ──────────────────────────────────────────────────────────

class TestPersonTracking:
    def test_person_detected_sets_present(self):
        sm = KitchenStateMachine()
        sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "person"}], {"zone_flame_status": {}}, [])
        assert sm.person_present is True
        assert sm.person_absent_since is None

    def test_person_leaves_sets_absent(self):
        sm = KitchenStateMachine()
        # Person present first
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], {"zone_flame_status": {}}, [])
        # Person gone
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], {"zone_flame_status": {}}, [])
        assert sm.person_present is False
        assert sm.person_absent_since == 1010.0

    def test_person_returns_clears_absence(self):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], {"zone_flame_status": {}}, [])
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], {"zone_flame_status": {}}, [])
        assert sm.person_absent_since is not None
        with patch("state_machine.time.time", return_value=1020.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], {"zone_flame_status": {}}, [])
        assert sm.person_present is True
        assert sm.person_absent_since is None


# ── Escalation timeline ─────────────────────────────────────────────────────

class TestEscalation:
    @pytest.fixture
    def zone(self):
        return [{"name": "Burner1", "x": 0, "y": 0, "w": 100, "h": 100}]

    def _heuristics(self, flame_on=True, zone_name="Burner1"):
        return {"zone_flame_status": {zone_name: flame_on}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}

    def test_flame_on_attended_no_alert(self, zone):
        sm = KitchenStateMachine()
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update(person, self._heuristics(), zone)
        assert alerts == []
        assert sm.zone_states["Burner1"] == BurnerState.ACTIVE_ATTENDED

    def test_flame_on_unattended_below_info_threshold(self, zone):
        sm = KitchenStateMachine()
        # Person leaves
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            alerts = sm.update([], self._heuristics(), zone)
        assert alerts == []
        assert sm.zone_states["Burner1"] == BurnerState.ACTIVE_UNATTENDED

    def test_escalation_to_info(self, zone):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        # 61 seconds after absence
        with patch("state_machine.time.time", return_value=1071.0):
            alerts = sm.update([], self._heuristics(), zone)
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.INFO
        assert sm.zone_states["Burner1"] == BurnerState.ALERT_INFO

    def test_escalation_to_warning(self, zone):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        # 181s after absence
        with patch("state_machine.time.time", return_value=1191.0):
            alerts = sm.update([], self._heuristics(), zone)
        assert any(a.severity == Severity.WARNING for a in alerts)
        assert sm.zone_states["Burner1"] == BurnerState.ALERT_WARNING

    def test_escalation_to_critical(self, zone):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        # 301s after absence
        with patch("state_machine.time.time", return_value=1311.0):
            alerts = sm.update([], self._heuristics(), zone)
        assert any(a.severity == Severity.CRITICAL for a in alerts)
        assert sm.zone_states["Burner1"] == BurnerState.ALERT_CRITICAL

    def test_critical_repeats_with_shorter_cooldown(self, zone):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1311.0):
            alerts1 = sm.update([], self._heuristics(), zone)
        assert len(alerts1) == 1
        # 16s later (> CRITICAL_REPEAT_SECONDS=15), should fire again
        with patch("state_machine.time.time", return_value=1327.0):
            alerts2 = sm.update([], self._heuristics(), zone)
        assert len(alerts2) == 1
        assert alerts2[0].severity == Severity.CRITICAL


# ── De-escalation ────────────────────────────────────────────────────────────

class TestDeescalation:
    @pytest.fixture
    def zone(self):
        return [{"name": "B1", "x": 0, "y": 0, "w": 100, "h": 100}]

    def _heuristics(self, flame_on=True):
        return {"zone_flame_status": {"B1": flame_on}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}

    def test_person_returns_deescalates(self, zone):
        sm = KitchenStateMachine()
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1200.0):
            sm.update([], self._heuristics(), zone)
        assert sm.zone_states["B1"] == BurnerState.ALERT_WARNING
        # Person returns
        with patch("state_machine.time.time", return_value=1210.0):
            sm.update(person, self._heuristics(), zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_ATTENDED

    def test_flame_off_resets_to_off(self, zone):
        sm = KitchenStateMachine()
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update([], self._heuristics(flame_on=True), zone)
        with patch("state_machine.time.time", return_value=1100.0):
            sm.update([], self._heuristics(flame_on=False), zone)
        assert sm.zone_states["B1"] == BurnerState.OFF


# ── Cooldown deduplication ───────────────────────────────────────────────────

class TestCooldowns:
    @pytest.fixture
    def zone(self):
        return [{"name": "B1", "x": 0, "y": 0, "w": 100, "h": 100}]

    def _heuristics(self, flame_on=True):
        return {"zone_flame_status": {"B1": flame_on}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}

    def test_info_alert_not_repeated_within_cooldown(self, zone):
        sm = KitchenStateMachine()
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        # First info alert at t=1071
        with patch("state_machine.time.time", return_value=1071.0):
            alerts1 = sm.update([], self._heuristics(), zone)
        assert len(alerts1) == 1
        # Same time range, within 30s cooldown
        with patch("state_machine.time.time", return_value=1080.0):
            alerts2 = sm.update([], self._heuristics(), zone)
        assert len(alerts2) == 0

    def test_alert_fires_after_cooldown_expires(self, zone):
        sm = KitchenStateMachine()
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], self._heuristics(), zone)
        with patch("state_machine.time.time", return_value=1071.0):
            alerts1 = sm.update([], self._heuristics(), zone)
        assert len(alerts1) == 1
        # 31s later, cooldown expired, still in info range
        with patch("state_machine.time.time", return_value=1102.0):
            alerts2 = sm.update([], self._heuristics(), zone)
        assert len(alerts2) == 1


# ── Hazard alerts ────────────────────────────────────────────────────────────

class TestHazardAlerts:
    def test_proximity_alert(self):
        sm = KitchenStateMachine()
        h = {
            "zone_flame_status": {},
            "proximity_alerts": [{"object": "bottle", "zone": "B1", "distance": 80}],
            "boilover_zones": [],
            "smoke_regions": [],
            "flames": [],
        }
        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update([], h, [])
        assert len(alerts) == 1
        assert alerts[0].type == "proximity"
        assert alerts[0].severity == Severity.WARNING

    def test_boilover_alert(self):
        sm = KitchenStateMachine()
        h = {
            "zone_flame_status": {},
            "proximity_alerts": [],
            "boilover_zones": ["B1"],
            "smoke_regions": [],
            "flames": [],
        }
        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update([], h, [])
        assert len(alerts) == 1
        assert alerts[0].type == "boilover"

    def test_smoke_alert_is_critical(self):
        sm = KitchenStateMachine()
        h = {
            "zone_flame_status": {},
            "proximity_alerts": [],
            "boilover_zones": [],
            "smoke_regions": [(100, 100, 50, 50)],
            "flames": [],
        }
        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update([], h, [])
        assert len(alerts) == 1
        assert alerts[0].severity == Severity.CRITICAL

    def test_assume_burners_active_escalates_without_flame(self):
        import config
        config.ASSUME_BURNERS_ACTIVE = True
        sm = KitchenStateMachine()
        zone = [{"name": "B1", "x": 0, "y": 0, "w": 100, "h": 100}]
        h_no_flame = {"zone_flame_status": {"B1": False}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        # Person present → attended
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, h_no_flame, zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_ATTENDED
        # Person leaves → should still escalate despite no flame
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], h_no_flame, zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_UNATTENDED

    def test_flame_with_no_zones_warns(self):
        sm = KitchenStateMachine()
        h = {
            "zone_flame_status": {},
            "proximity_alerts": [],
            "boilover_zones": [],
            "smoke_regions": [],
            "flames": [(100, 100, 50, 50)],
        }
        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update([], h, [])
        assert len(alerts) == 1
        assert alerts[0].type == "flame"
        assert alerts[0].severity == Severity.WARNING


# ── get_status ───────────────────────────────────────────────────────────────

class TestGetStatus:
    def test_status_empty(self):
        sm = KitchenStateMachine()
        status = sm.get_status([])
        assert status["person_present"] is False
        assert status["zones"] == {}

    def test_status_with_zone(self):
        sm = KitchenStateMachine()
        zone = [{"name": "B1", "x": 0, "y": 0, "w": 100, "h": 100}]
        h = {"zone_flame_status": {"B1": True}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, h, zone)
        with patch("state_machine.time.time", return_value=1005.0):
            status = sm.get_status(zone)
        assert status["person_present"] is True
        assert "B1" in status["zones"]
        assert status["zones"]["B1"]["state"] == "active_attended"

    def test_status_absence_seconds(self):
        sm = KitchenStateMachine()
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, {"zone_flame_status": {}}, [])
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], {"zone_flame_status": {}}, [])
        with patch("state_machine.time.time", return_value=1030.0):
            status = sm.get_status([])
        assert status["person_present"] is False
        assert status["person_absent_seconds"] == pytest.approx(20.0, abs=1)
