"""Tests for main.py entry point."""

from unittest.mock import patch

import config
from main import parse_args, SafeFlame


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
