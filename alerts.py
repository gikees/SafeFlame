"""Voice alert system for SafeFlame — TTS with cooldowns and background threading."""

import subprocess
import sys
import threading
import time

import config
from state_machine import Alert, Severity


class AlertManager:
    """Manages voice alerts with cooldowns and background TTS."""

    def __init__(self):
        self._tts_engine = None
        self._tts_lock = threading.Lock()
        self._init_tts()

        # Alert log (for dashboard)
        self.alert_log: list[dict] = []
        self._log_lock = threading.Lock()

    def _init_tts(self):
        """Initialize pyttsx3, with fallback flag."""
        try:
            import pyttsx3
            self._tts_engine = pyttsx3.init()
            self._tts_engine.setProperty("rate", config.TTS_RATE)
        except Exception:
            self._tts_engine = None

    def handle_alert(self, alert: Alert, advice: str | None = None):
        """Process an alert: log it and speak it in a background thread."""
        message = alert.message
        if advice:
            message += f" {advice}"

        # Log
        entry = alert.to_dict()
        entry["advice"] = advice
        with self._log_lock:
            self.alert_log.append(entry)
            # Keep last 100 alerts
            if len(self.alert_log) > 100:
                self.alert_log = self.alert_log[-100:]

        # Speak in background
        thread = threading.Thread(
            target=self._speak, args=(message, alert.severity), daemon=True
        )
        thread.start()

    def _speak(self, text: str, severity: Severity):
        """Speak text using TTS engine or system fallback."""
        volume = {
            Severity.INFO: config.TTS_VOLUME_INFO,
            Severity.WARNING: config.TTS_VOLUME_WARNING,
            Severity.CRITICAL: config.TTS_VOLUME_CRITICAL,
        }.get(severity, config.TTS_VOLUME_INFO)

        with self._tts_lock:
            if self._tts_engine is not None:
                try:
                    self._tts_engine.setProperty("volume", volume)
                    self._tts_engine.say(text)
                    self._tts_engine.runAndWait()
                    return
                except Exception:
                    pass

            # Fallback: system TTS
            self._system_speak(text)

    def _system_speak(self, text: str):
        """Fallback TTS using system commands."""
        safe_text = text.replace('"', '\\"')
        if sys.platform == "darwin":
            subprocess.run(["say", safe_text], timeout=15, check=False)
        elif sys.platform == "linux":
            subprocess.run(
                ["espeak", safe_text], timeout=15, check=False,
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
            )

    def get_recent_alerts(self, n: int = 50) -> list[dict]:
        """Return the most recent n alerts."""
        with self._log_lock:
            return list(self.alert_log[-n:])
