"""Tests for HeuristicDetector — flame, smoke, boil-over, proximity, overlays."""

from unittest.mock import patch

import cv2
import numpy as np
import pytest

import config
from heuristics import HeuristicDetector


class TestFlameDetection:
    def test_no_flame_on_black_frame(self, black_frame):
        hd = HeuristicDetector()
        result = hd.analyze(black_frame, [], [])
        assert result["flames"] == []

    def test_flame_detected_on_flame_frame(self, flame_frame):
        hd = HeuristicDetector()
        # Lower threshold to ensure detection
        config.FLAME_MIN_CONTOUR_AREA = 50
        result = hd.analyze(flame_frame, [], [])
        assert len(result["flames"]) > 0

    def test_flame_below_min_area_filtered(self, black_frame):
        hd = HeuristicDetector()
        # Place a tiny flame pixel — won't meet 500px² threshold
        black_frame[100, 100] = (0, 100, 255)
        result = hd.analyze(black_frame, [], [])
        assert result["flames"] == []


class TestSmokeDetection:
    def test_no_smoke_on_black_frame(self, black_frame):
        hd = HeuristicDetector()
        result = hd.analyze(black_frame, [], [])
        assert result["smoke_regions"] == []

    def test_smoke_requires_persistence(self, smoke_frame):
        hd = HeuristicDetector()
        config.SMOKE_MIN_CONTOUR_AREA = 50
        # First call — smoke seen but not persisted
        with patch("heuristics.time.time", return_value=1000.0):
            r1 = hd.analyze(smoke_frame, [], [])
        assert r1["smoke_regions"] == []
        # Second call within persistence window
        with patch("heuristics.time.time", return_value=1001.0):
            r2 = hd.analyze(smoke_frame, [], [])
        assert r2["smoke_regions"] == []
        # After persistence threshold
        with patch("heuristics.time.time", return_value=1004.0):
            r3 = hd.analyze(smoke_frame, [], [])
        assert len(r3["smoke_regions"]) > 0

    def test_smoke_stale_entries_cleaned(self, smoke_frame, black_frame):
        hd = HeuristicDetector()
        config.SMOKE_MIN_CONTOUR_AREA = 50
        with patch("heuristics.time.time", return_value=1000.0):
            hd.analyze(smoke_frame, [], [])
        assert len(hd._smoke_first_seen) > 0
        # Black frame clears smoke
        with patch("heuristics.time.time", return_value=1010.0):
            hd.analyze(black_frame, [], [])
        assert len(hd._smoke_first_seen) == 0


class TestBoiloverDetection:
    def test_no_boilover_without_prev_frame(self, black_frame, sample_burner_zones):
        hd = HeuristicDetector()
        result = hd.analyze(black_frame, [], sample_burner_zones)
        assert result["boilover_zones"] == []

    def test_no_boilover_on_static_frames(self, black_frame, sample_burner_zones):
        hd = HeuristicDetector()
        hd.analyze(black_frame, [], sample_burner_zones)
        result = hd.analyze(black_frame, [], sample_burner_zones)
        assert result["boilover_zones"] == []


class TestZoneFlameOverlap:
    def test_flame_overlapping_zone(self, sample_burner_zones):
        hd = HeuristicDetector()
        flames = [(150, 150, 50, 50)]  # overlaps Front Left zone
        status = hd._check_zones_for_flame(flames, sample_burner_zones)
        assert status["Front Left"] is True
        assert status["Front Right"] is False

    def test_flame_not_overlapping_any_zone(self, sample_burner_zones):
        hd = HeuristicDetector()
        flames = [(800, 800, 20, 20)]
        status = hd._check_zones_for_flame(flames, sample_burner_zones)
        assert status["Front Left"] is False
        assert status["Front Right"] is False


class TestProximityDetection:
    def test_bottle_near_zone_detected(self, sample_burner_zones, bottle_near_zone):
        hd = HeuristicDetector()
        alerts = hd._check_proximity([bottle_near_zone], sample_burner_zones)
        assert len(alerts) >= 1
        assert alerts[0]["object"] == "bottle"

    def test_non_flammable_not_flagged(self, sample_burner_zones):
        hd = HeuristicDetector()
        spoon = {"class": "spoon", "confidence": 0.8, "bbox": (180, 180, 220, 280), "label": "spoon"}
        alerts = hd._check_proximity([spoon], sample_burner_zones)
        assert alerts == []

    def test_far_object_not_flagged(self, sample_burner_zones):
        hd = HeuristicDetector()
        far_bottle = {"class": "bottle", "confidence": 0.8, "bbox": (900, 900, 950, 1000), "label": "bottle"}
        alerts = hd._check_proximity([far_bottle], sample_burner_zones)
        assert alerts == []


class TestDrawOverlays:
    def test_draw_overlays_returns_frame(self, black_frame, sample_burner_zones):
        hd = HeuristicDetector()
        heuristics = {
            "flames": [(150, 150, 30, 30)],
            "smoke_regions": [(50, 50, 40, 40)],
            "proximity_alerts": [{"object": "bottle", "zone": "Front Left", "distance": 80}],
            "zone_flame_status": {"Front Left": True, "Front Right": False},
            "boilover_zones": [],
        }
        result = hd.draw_overlays(black_frame, heuristics, sample_burner_zones)
        assert result.shape == black_frame.shape
        # Should not be all zeros (annotations were drawn)
        assert np.any(result != 0)

    def test_draw_overlays_does_not_modify_original(self, black_frame, sample_burner_zones):
        hd = HeuristicDetector()
        heuristics = {
            "flames": [], "smoke_regions": [], "proximity_alerts": [],
            "zone_flame_status": {}, "boilover_zones": [],
        }
        original = black_frame.copy()
        hd.draw_overlays(black_frame, heuristics, sample_burner_zones)
        np.testing.assert_array_equal(black_frame, original)
