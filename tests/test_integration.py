"""Integration tests — full pipeline scenarios with mocked externals."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config
from heuristics import HeuristicDetector
from state_machine import KitchenStateMachine, BurnerState, Severity


class TestAttendedCooking:
    """Person present + flame on → no alerts, stays ACTIVE_ATTENDED."""

    def test_no_escalation_while_attended(self):
        sm = KitchenStateMachine()
        hd = HeuristicDetector()
        zone = [{"name": "B1", "x": 100, "y": 100, "w": 200, "h": 200}]
        config.BURNER_ZONES = zone
        config.FLAME_MIN_CONTOUR_AREA = 50

        # Build a frame with a flame-colored patch inside the zone
        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        frame[150:200, 150:200] = (0, 100, 255)  # flame-like BGR

        person = [{"class": "person", "confidence": 0.9, "bbox": (300, 100, 500, 600), "label": "person 90%"}]

        with patch("state_machine.time.time", return_value=1000.0), \
             patch("heuristics.time.time", return_value=1000.0):
            heuristic_results = hd.analyze(frame, person, zone)
            alerts = sm.update(person, heuristic_results, zone)

        assert alerts == []
        assert sm.zone_states.get("B1") in (BurnerState.ACTIVE_ATTENDED, BurnerState.OFF)


class TestEscalationAndReturn:
    """Person leaves → escalates to WARNING → person returns → de-escalates."""

    def test_full_escalation_deescalation_cycle(self):
        sm = KitchenStateMachine()
        zone = [{"name": "B1", "x": 0, "y": 0, "w": 100, "h": 100}]
        person = [{"class": "person", "confidence": 0.9, "bbox": (0, 0, 1, 1), "label": "p"}]
        h_on = {"zone_flame_status": {"B1": True}, "proximity_alerts": [], "boilover_zones": [], "smoke_regions": [], "flames": []}

        # T=0: person present, flame on
        with patch("state_machine.time.time", return_value=1000.0):
            sm.update(person, h_on, zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_ATTENDED

        # T=10: person leaves
        with patch("state_machine.time.time", return_value=1010.0):
            sm.update([], h_on, zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_UNATTENDED

        # T=200: past warning threshold
        with patch("state_machine.time.time", return_value=1200.0):
            alerts = sm.update([], h_on, zone)
        assert sm.zone_states["B1"] == BurnerState.ALERT_WARNING
        assert any(a.severity == Severity.WARNING for a in alerts)

        # T=210: person returns
        with patch("state_machine.time.time", return_value=1210.0):
            alerts = sm.update(person, h_on, zone)
        assert sm.zone_states["B1"] == BurnerState.ACTIVE_ATTENDED
        # No unattended alerts when person is present
        assert not any(a.type == "unattended" for a in alerts)


class TestProximityPipeline:
    """Flammable object near burner → proximity alert from state machine."""

    def test_bottle_triggers_proximity_alert(self):
        sm = KitchenStateMachine()
        hd = HeuristicDetector()
        zone = [{"name": "B1", "x": 100, "y": 100, "w": 200, "h": 200}]
        config.BURNER_ZONES = zone

        frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        bottle = [{"class": "bottle", "confidence": 0.8, "bbox": (180, 180, 220, 280), "label": "bottle 80%"}]

        with patch("heuristics.time.time", return_value=1000.0):
            heuristic_results = hd.analyze(frame, bottle, zone)
        assert len(heuristic_results["proximity_alerts"]) >= 1

        with patch("state_machine.time.time", return_value=1000.0):
            alerts = sm.update(bottle, heuristic_results, zone)
        assert any(a.type == "proximity" for a in alerts)


class TestSmokePersistence:
    """Smoke must persist before triggering a CRITICAL alert."""

    def test_smoke_needs_persistence_for_critical(self):
        sm = KitchenStateMachine()
        hd = HeuristicDetector()
        config.SMOKE_MIN_CONTOUR_AREA = 50

        smoke_frame = np.zeros((720, 1280, 3), dtype=np.uint8)
        smoke_frame[100:200, 100:250] = (180, 180, 180)

        # T=0: first observation — not persisted yet
        with patch("heuristics.time.time", return_value=1000.0):
            r1 = hd.analyze(smoke_frame, [], [])
        with patch("state_machine.time.time", return_value=1000.0):
            a1 = sm.update([], r1, [])
        assert a1 == []  # no smoke confirmed

        # T=1: still within persistence window
        with patch("heuristics.time.time", return_value=1001.0):
            r2 = hd.analyze(smoke_frame, [], [])
        with patch("state_machine.time.time", return_value=1001.0):
            a2 = sm.update([], r2, [])
        assert a2 == []

        # T=4: past persistence threshold
        with patch("heuristics.time.time", return_value=1004.0):
            r3 = hd.analyze(smoke_frame, [], [])
        assert len(r3["smoke_regions"]) > 0
        with patch("state_machine.time.time", return_value=1004.0):
            a3 = sm.update([], r3, [])
        assert any(a.type == "smoke" and a.severity == Severity.CRITICAL for a in a3)


class TestAlertDeduplication:
    """Same alert type+zone within cooldown should not fire twice."""

    def test_proximity_dedup(self):
        sm = KitchenStateMachine()
        h = {
            "zone_flame_status": {},
            "proximity_alerts": [{"object": "bottle", "zone": "B1", "distance": 80}],
            "boilover_zones": [],
            "smoke_regions": [],
            "flames": [],
        }
        with patch("state_machine.time.time", return_value=1000.0):
            a1 = sm.update([], h, [])
        assert len(a1) == 1

        # Within cooldown
        with patch("state_machine.time.time", return_value=1010.0):
            a2 = sm.update([], h, [])
        assert len(a2) == 0

        # After cooldown
        with patch("state_machine.time.time", return_value=1031.0):
            a3 = sm.update([], h, [])
        assert len(a3) == 1
