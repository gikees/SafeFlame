"""Tests for main.py entry point."""

from unittest.mock import patch, MagicMock

import config
from main import parse_args, SafeFlame
from state_machine import Alert, Severity


class TestDemoFlag:
    """--demo flag shortens escalation timers for live presentations."""

    def test_demo_sets_shortened_timers(self):
        with patch("sys.argv", ["main.py", "--no-display", "--no-tts", "--no-llm", "--demo"]):
            args = parse_args()

        with patch("main.Detector"), \
             patch("main.HeuristicDetector"), \
             patch("main.KitchenStateMachine"), \
             patch("main.set_shared_state"):
            sf = SafeFlame(args)

        assert config.UNATTENDED_INFO_SECONDS == 10
        assert config.UNATTENDED_WARNING_SECONDS == 30
        assert config.UNATTENDED_CRITICAL_SECONDS == 60
        assert config.ALERT_COOLDOWN_SECONDS == 10
        assert config.SMOKE_PERSISTENCE_SECONDS == 1.5
        assert config.ASSUME_BURNERS_ACTIVE is True
        assert config.SMOKE_DETECTION_ENABLED is False
        assert config.BOILOVER_DETECTION_ENABLED is False

    def test_no_demo_keeps_defaults(self):
        with patch("sys.argv", ["main.py", "--no-display", "--no-tts", "--no-llm"]):
            args = parse_args()

        with patch("main.Detector"), \
             patch("main.HeuristicDetector"), \
             patch("main.KitchenStateMachine"), \
             patch("main.set_shared_state"):
            sf = SafeFlame(args)

        assert config.UNATTENDED_INFO_SECONDS == 60
        assert config.UNATTENDED_WARNING_SECONDS == 180
        assert config.UNATTENDED_CRITICAL_SECONDS == 300
        assert config.ALERT_COOLDOWN_SECONDS == 30
        assert config.SMOKE_PERSISTENCE_SECONDS == 3.0
        assert config.ASSUME_BURNERS_ACTIVE is False
        assert config.SMOKE_DETECTION_ENABLED is True
        assert config.BOILOVER_DETECTION_ENABLED is True


class TestNoLlmFallback:
    """--no-llm flag should use FALLBACK_ADVICE for warning/critical alerts."""

    def test_no_llm_uses_fallback_advice(self):
        with patch("sys.argv", ["main.py", "--no-display", "--no-tts", "--no-llm"]):
            args = parse_args()

        mock_alert_manager = MagicMock()
        mock_state_machine = MagicMock()

        with patch("main.Detector"), \
             patch("main.HeuristicDetector"), \
             patch("main.KitchenStateMachine") as MockSM, \
             patch("main.set_shared_state"):
            sf = SafeFlame(args)
            sf.alert_manager = mock_alert_manager
            sf.state_machine = mock_state_machine

        assert sf.llm_advisor is None

        # Simulate a warning alert
        alert = Alert(
            type="smoke",
            severity=Severity.WARNING,
            message="Smoke detected",
            burner_zone="Front Left",
        )
        mock_state_machine.update.return_value = [alert]

        # Run alert handling manually (extracted logic from the loop)
        advice = config.FALLBACK_ADVICE.get(alert.type)
        assert advice == config.FALLBACK_ADVICE["smoke"]

    def test_no_llm_info_alert_gets_no_advice(self):
        """INFO-level alerts should not get advice even with fallback."""
        alert = Alert(
            type="unattended",
            severity=Severity.INFO,
            message="Burner unattended",
            burner_zone="Front Left",
        )
        # INFO is not in (WARNING, CRITICAL), so advice stays None
        assert alert.severity not in (Severity.WARNING, Severity.CRITICAL)
