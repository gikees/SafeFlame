"""Tests for LLMAdvisor — caching, timeouts, unavailable fallback."""

import sys
import time as _time
import threading
from unittest.mock import MagicMock, patch

import pytest

import config
from llm_advisor import LLMAdvisor, HAZARD_PROMPTS


def _make_advisor(available=True, chat_return="Turn off the stove.", chat_side_effect=None):
    """Helper to create an LLMAdvisor with a controlled mock client."""
    mock_ollama = MagicMock()
    if not available:
        mock_ollama.Client.side_effect = Exception("No Ollama")
    else:
        mock_client = MagicMock()
        if chat_side_effect:
            mock_client.chat.side_effect = chat_side_effect
        else:
            mock_client.chat.return_value = {"message": {"content": chat_return}}
        mock_ollama.Client.return_value = mock_client

    # Temporarily replace the ollama mock in sys.modules
    old = sys.modules.get("ollama")
    sys.modules["ollama"] = mock_ollama
    try:
        advisor = LLMAdvisor()
    finally:
        if old is not None:
            sys.modules["ollama"] = old
    return advisor


class TestLLMAvailability:
    def test_unavailable_returns_none(self):
        advisor = _make_advisor(available=False)
        assert advisor._available is False
        result = advisor.get_safety_advice("smoke")
        assert result is None

    def test_available_when_client_works(self):
        advisor = _make_advisor(available=True)
        assert advisor._available is True


class TestCaching:
    def test_cached_response_reused(self):
        advisor = _make_advisor(chat_return="Turn off the stove.")
        r1 = advisor.get_safety_advice("smoke")
        assert r1 == "Turn off the stove."
        # Second call should use cache — verify by checking _client.chat call count
        advisor._client.chat.reset_mock()
        r2 = advisor.get_safety_advice("smoke")
        assert r2 == "Turn off the stove."
        advisor._client.chat.assert_not_called()

    def test_different_context_not_cached(self):
        advisor = _make_advisor(chat_return="Safety tip.")
        advisor.get_safety_advice("proximity", {"object": "bottle", "zone": "B1"})
        advisor._client.chat.reset_mock()
        advisor._client.chat.return_value = {"message": {"content": "Different tip."}}
        advisor.get_safety_advice("proximity", {"object": "cup", "zone": "B2"})
        advisor._client.chat.assert_called_once()

    def test_cache_size_limited(self):
        advisor = _make_advisor(chat_return="Tip.")
        for i in range(105):
            advisor.get_safety_advice(f"hazard_{i}")
        assert len(advisor._cache) <= 101


class TestTimeout:
    def test_slow_response_returns_none(self):
        def slow_chat(**kwargs):
            _time.sleep(5)
            return {"message": {"content": "Too slow."}}

        advisor = _make_advisor(chat_side_effect=slow_chat)
        config.LLM_TIMEOUT_SECONDS = 0.1
        result = advisor.get_safety_advice("smoke")
        assert result is None


class TestExceptionHandling:
    def test_chat_exception_returns_none(self):
        advisor = _make_advisor(chat_side_effect=RuntimeError("Model crashed"))
        result = advisor.get_safety_advice("smoke")
        assert result is None

    def test_hazard_prompts_formatting(self):
        prompt = HAZARD_PROMPTS["proximity"]
        formatted = prompt.format(object="bottle", zone="B1")
        assert "bottle" in formatted
        assert "B1" in formatted

    def test_unknown_hazard_uses_fallback_prompt(self):
        advisor = _make_advisor(chat_return="Generic advice.")
        result = advisor.get_safety_advice("unknown_hazard_type")
        assert result == "Generic advice."
