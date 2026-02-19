"""Tests for AlertManager — logging, TTS, thread safety."""

import threading
import time

from unittest.mock import MagicMock, patch

import pytest

from alerts import AlertManager
from state_machine import Alert, Severity


class TestAlertLogging:
    def test_alert_logged(self):
        am = AlertManager()
        am._tts_engine = None  # skip TTS to avoid side-effects
        alert = Alert(type="smoke", severity=Severity.CRITICAL, message="Smoke!", timestamp=1000.0)
        am.handle_alert(alert)
        time.sleep(0.1)
        assert len(am.alert_log) == 1
        assert am.alert_log[0]["type"] == "smoke"

    def test_alert_logged_with_advice(self):
        am = AlertManager()
        am._tts_engine = None
        alert = Alert(type="smoke", severity=Severity.CRITICAL, message="Smoke!", timestamp=1000.0)
        am.handle_alert(alert, advice="Leave the kitchen")
        time.sleep(0.1)
        assert am.alert_log[0]["advice"] == "Leave the kitchen"

    def test_log_capped_at_100(self):
        am = AlertManager()
        am._tts_engine = None
        for i in range(110):
            alert = Alert(type="test", severity=Severity.INFO, message=f"Alert {i}", timestamp=float(i))
            am.handle_alert(alert)
        time.sleep(0.3)
        assert len(am.alert_log) <= 100

    def test_get_recent_alerts(self):
        am = AlertManager()
        am._tts_engine = None
        for i in range(10):
            alert = Alert(type="test", severity=Severity.INFO, message=f"Alert {i}", timestamp=float(i))
            am.handle_alert(alert)
        time.sleep(0.1)
        recent = am.get_recent_alerts(5)
        assert len(recent) == 5


class TestTTS:
    def test_tts_failure_graceful(self):
        am = AlertManager()
        # Force TTS to raise
        am._tts_engine = MagicMock()
        am._tts_engine.runAndWait.side_effect = RuntimeError("TTS broken")
        alert = Alert(type="test", severity=Severity.INFO, message="Test", timestamp=1000.0)
        # Should not raise — falls back to system speak
        with patch.object(am, "_system_speak"):
            am.handle_alert(alert)
        time.sleep(0.2)

    def test_severity_volume_mapping_critical(self):
        am = AlertManager()
        mock_engine = MagicMock()
        am._tts_engine = mock_engine
        am._speak("test", Severity.CRITICAL)
        mock_engine.setProperty.assert_any_call("volume", 1.0)

    def test_severity_volume_mapping_info(self):
        am = AlertManager()
        mock_engine = MagicMock()
        am._tts_engine = mock_engine
        am._speak("test", Severity.INFO)
        mock_engine.setProperty.assert_any_call("volume", 0.7)

    def test_no_tts_engine_uses_system_speak(self):
        am = AlertManager()
        am._tts_engine = None
        with patch.object(am, "_system_speak") as mock_sys:
            am._speak("hello", Severity.WARNING)
        mock_sys.assert_called_once_with("hello")


class TestThreadSafety:
    def test_concurrent_alert_logging(self):
        am = AlertManager()
        am._tts_engine = None

        def log_alerts():
            for i in range(20):
                alert = Alert(type="test", severity=Severity.INFO, message=f"t-{i}", timestamp=float(i))
                am.handle_alert(alert)

        threads = [threading.Thread(target=log_alerts) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        time.sleep(0.3)
        # All 60 alerts should be logged (capped at 100)
        assert len(am.alert_log) == 60
