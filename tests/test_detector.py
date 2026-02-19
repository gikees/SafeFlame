"""Tests for Detector — mocked YOLO inference."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

import config


class TestDetector:
    @pytest.fixture
    def mock_yolo(self):
        """Set up a mock YOLO model with fake detections."""
        mock_model = MagicMock()
        mock_model.names = {0: "person", 1: "car", 2: "bottle"}

        # Create a mock result box
        mock_box = MagicMock()
        mock_box.cls = [MagicMock(__getitem__=lambda s, i: 0)]  # cls_id = 0
        mock_box.cls[0].__int__ = lambda s: 0
        mock_box.conf = [MagicMock(__getitem__=lambda s, i: 0.95)]
        mock_box.conf[0].__float__ = lambda s: 0.95
        mock_box.xyxy = [MagicMock()]
        mock_box.xyxy[0].tolist = lambda: [100.0, 50.0, 300.0, 400.0]

        mock_result = MagicMock()
        mock_result.boxes = [mock_box]
        mock_model.return_value = [mock_result]

        return mock_model

    def test_detect_returns_person(self, mock_yolo, black_frame):
        with patch("detector.YOLO", return_value=mock_yolo):
            from detector import Detector
            det = Detector()
            det.model = mock_yolo
            results = det.detect(black_frame)
        assert len(results) == 1
        assert results[0]["class"] == "person"
        assert results[0]["confidence"] == 0.95
        assert results[0]["bbox"] == (100, 50, 300, 400)

    def test_detect_filters_non_interest_classes(self, mock_yolo, black_frame):
        # Change class to "car" which is not in YOLO_CLASSES_OF_INTEREST
        mock_box = mock_yolo.return_value[0].boxes[0]
        mock_box.cls = [MagicMock()]
        mock_box.cls[0].__int__ = lambda s: 1  # car
        mock_box.cls[0].__getitem__ = lambda s, i: MagicMock(__int__=lambda s2: 1)

        with patch("detector.YOLO", return_value=mock_yolo):
            from detector import Detector
            det = Detector()
            det.model = mock_yolo
            # Need to fix cls to return int 1
            mock_box.cls[0] = MagicMock()
            int_mock = MagicMock(return_value=1)
            mock_box.cls = [int_mock]
            results = det.detect(black_frame)
        assert len(results) == 0

    def test_draw_returns_annotated_frame(self, black_frame):
        with patch("detector.YOLO"):
            from detector import Detector
            det = Detector()
        detections = [{"class": "person", "confidence": 0.9, "bbox": (50, 50, 200, 300), "label": "person 90%"}]
        annotated = det.draw(black_frame, detections)
        assert annotated.shape == black_frame.shape
        assert np.any(annotated != black_frame)

    def test_draw_does_not_modify_original(self, black_frame):
        with patch("detector.YOLO"):
            from detector import Detector
            det = Detector()
        original = black_frame.copy()
        det.draw(black_frame, [{"class": "person", "confidence": 0.9, "bbox": (50, 50, 200, 300), "label": "person 90%"}])
        np.testing.assert_array_equal(black_frame, original)

    def test_detect_empty_results(self, mock_yolo, black_frame):
        mock_yolo.return_value = [MagicMock(boxes=[])]
        with patch("detector.YOLO", return_value=mock_yolo):
            from detector import Detector
            det = Detector()
            det.model = mock_yolo
            results = det.detect(black_frame)
        assert results == []
