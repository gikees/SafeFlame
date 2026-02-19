"""Local Ollama LLM integration for contextual safety advice."""

import threading
import time
from functools import lru_cache

import config


# Pre-built system prompt
SYSTEM_PROMPT = (
    "You are a kitchen safety advisor. Give ONE short, actionable safety tip "
    "(one sentence, under 30 words). Be direct and specific."
)

# Pre-built hazard prompts
HAZARD_PROMPTS = {
    "unattended": "A kitchen burner has been left on with no one watching. What should the person do?",
    "proximity": "A flammable object ({object}) is near an active burner ({zone}). What's the immediate risk and action?",
    "boilover": "A pot is boiling over on burner {zone}. What should the person do right now?",
    "smoke": "Smoke has been detected in the kitchen. What's the safest immediate action?",
    "flame": "An open flame has been detected in the kitchen. What should be done?",
    "grease_fire": "A grease fire may be starting on the stove. What's the correct response?",
}


class LLMAdvisor:
    """Queries a local Ollama LLM for safety advice with caching and timeouts."""

    def __init__(self):
        self._client = None
        self._available = False
        # Cache for recent advice
        self._cache: dict[str, str] = {}
        self._cache_lock = threading.Lock()
        # Init Ollama in background so startup is never blocked
        self._init_thread = threading.Thread(target=self._init_client, daemon=True)
        self._init_thread.start()
        self._init_thread.join(timeout=config.LLM_TIMEOUT_SECONDS)

    def _init_client(self):
        """Try to connect to Ollama."""
        try:
            import ollama
            self._client = ollama.Client(host=config.OLLAMA_URL)
            # Quick check if model is available
            self._client.list()
            self._available = True
        except Exception:
            self._available = False

    def get_safety_advice(
        self, hazard_type: str, context: dict | None = None
    ) -> str | None:
        """Get a one-sentence safety tip for the given hazard.

        Falls back to config.FALLBACK_ADVICE if LLM is unavailable or times out.
        """
        if not self._available:
            return config.FALLBACK_ADVICE.get(hazard_type)

        # Build the cache key
        cache_key = hazard_type
        if context:
            cache_key += "_" + "_".join(f"{k}={v}" for k, v in sorted(context.items()))

        # Check cache
        with self._cache_lock:
            if cache_key in self._cache:
                return self._cache[cache_key]

        # Build prompt
        prompt_template = HAZARD_PROMPTS.get(hazard_type, f"Kitchen hazard: {hazard_type}. What should be done?")
        if context:
            prompt = prompt_template.format(**context)
        else:
            prompt = prompt_template

        # Query with timeout
        result = [None]

        def _query():
            try:
                response = self._client.chat(
                    model=config.OLLAMA_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                result[0] = response["message"]["content"].strip()
            except Exception:
                pass

        thread = threading.Thread(target=_query, daemon=True)
        thread.start()
        thread.join(timeout=config.LLM_TIMEOUT_SECONDS)

        advice = result[0]
        if advice:
            with self._cache_lock:
                self._cache[cache_key] = advice
                # Limit cache size
                if len(self._cache) > 100:
                    oldest = next(iter(self._cache))
                    del self._cache[oldest]
            return advice

        # LLM timed out or errored — use fallback
        return config.FALLBACK_ADVICE.get(hazard_type)
