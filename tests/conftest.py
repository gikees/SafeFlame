"""Shared fixtures for SafeFlame tests."""

import sys
from pathlib import Path
from unittest.mock import MagicMock

import pytest
import numpy as np

# Ensure project root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ── Mock heavy third-party imports before any project module loads ────────────
# ultralytics
_mock_yolo_module = MagicMock()
sys.modules["ultralytics"] = _mock_yolo_module

# pyttsx3 — imported locally inside alerts._init_tts()
_mock_pyttsx3 = MagicMock()
sys.modules["pyttsx3"] = _mock_pyttsx3

# ollama — imported locally inside llm_advisor._init_client()
_mock_ollama = MagicMock()
sys.modules["ollama"] = _mock_ollama

# uvicorn (not needed at import time for tests)
sys.modules.setdefault("uvicorn", MagicMock())

import config  # noqa: E402 — must come after sys.path tweak


@pytest.fixture(autouse=True)
def _reset_config():
    """Reset config values to defaults before each test."""
    config.BURNER_ZONES = []
    config.UNATTENDED_INFO_SECONDS = 60
    config.UNATTENDED_WARNING_SECONDS = 180
    config.UNATTENDED_CRITICAL_SECONDS = 300
    config.ALERT_COOLDOWN_SECONDS = 30
    config.CRITICAL_REPEAT_SECONDS = 15
    config.FLAME_MIN_CONTOUR_AREA = 500
    config.SMOKE_MIN_CONTOUR_AREA = 2000
    config.SMOKE_PERSISTENCE_SECONDS = 3.0
    config.PROXIMITY_DISTANCE_PX = 150
    config.FLAMMABLE_OBJECTS = ["bottle", "cup", "cell phone", "book", "paper"]
    config.TTS_RATE = 175
    config.TTS_VOLUME_INFO = 0.7
    config.TTS_VOLUME_WARNING = 0.9
    config.TTS_VOLUME_CRITICAL = 1.0
    config.YOLO_CONFIDENCE = 0.4
    config.YOLO_CLASSES_OF_INTEREST = [
        "person", "oven", "microwave", "toaster", "knife",
        "cup", "bowl", "bottle", "fork", "spoon",
        "sink", "refrigerator", "cell phone",
    ]
    config.LLM_TIMEOUT_SECONDS = 3.0
    config.OLLAMA_MODEL = "llama3.2:3b"
    yield


@pytest.fixture
def black_frame():
    """720p black frame."""
    return np.zeros((720, 1280, 3), dtype=np.uint8)


@pytest.fixture
def flame_frame():
    """Frame with a bright orange-red region that triggers flame detection."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Draw a flame-colored rectangle (BGR → high V, high S, low H in HSV)
    # Orange-red in BGR: B=0, G=100, R=255
    frame[300:360, 500:560] = (0, 100, 255)
    return frame


@pytest.fixture
def smoke_frame():
    """Frame with a gray region that triggers smoke detection."""
    frame = np.zeros((720, 1280, 3), dtype=np.uint8)
    # Gray smoke-like region: BGR (180, 180, 180)
    frame[100:200, 100:250] = (180, 180, 180)
    return frame


@pytest.fixture
def sample_burner_zones():
    """Two burner zones for testing."""
    return [
        {"name": "Front Left", "x": 100, "y": 100, "w": 200, "h": 200},
        {"name": "Front Right", "x": 400, "y": 100, "w": 200, "h": 200},
    ]


@pytest.fixture
def person_detection():
    """Single person detection dict."""
    return {
        "class": "person",
        "confidence": 0.92,
        "bbox": (100, 50, 300, 500),
        "label": "person 92%",
    }


@pytest.fixture
def bottle_near_zone():
    """A bottle detection close to Front Left zone center (200, 200)."""
    return {
        "class": "bottle",
        "confidence": 0.75,
        "bbox": (180, 180, 220, 280),
        "label": "bottle 75%",
    }
