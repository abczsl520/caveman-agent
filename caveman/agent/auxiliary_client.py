"""Auxiliary Client — secondary LLM client for summarization and analysis.

Provides a lightweight LLM client for auxiliary tasks (summarization,
classification, analysis) that doesn't use the primary model's context.
Extracted from Hermes agent/auxiliary_client.py (2613 lines).
"""
from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

__all__ = [
    "AuxiliaryConfig",
    "call_llm",
    "summarize",
    "classify",
    "translate",
]


logger = logging.getLogger("caveman.agent.auxiliary_client")


@dataclass
class AuxiliaryConfig:
    """Configuration for the auxiliary LLM client."""
    provider: str = ""  # openai | anthropic | deepseek | auto
    model: str = ""
    api_key: str = ""
    base_url: str = ""
    max_tokens: int = 4096
    temperature: float = 0.3
    timeout: float = 60.0

    @classmethod
    def from_env(cls, task: str = "") -> "AuxiliaryConfig":
        """Resolve config from environment variables."""
        config = cls()

        # Task-specific override: CAVEMAN_AUX_{TASK}_MODEL
        task_upper = task.upper().replace("-", "_") if task else ""
        if task_upper:
            config.model = os.environ.get(f"CAVEMAN_AUX_{task_upper}_MODEL", "")
            config.provider = os.environ.get(f"CAVEMAN_AUX_{task_upper}_PROVIDER", "")

        # General auxiliary config
        if not config.model:
            config.model = os.environ.get("CAVEMAN_AUX_MODEL", "")
        if not config.provider:
            config.provider = os.environ.get("CAVEMAN_AUX_PROVIDER", "")

        # Auto-resolve provider from available keys
        if not config.provider:
            config.provider = _auto_detect_provider()

        # Resolve API key and base URL
        config.api_key, config.base_url = _resolve_credentials(config.provider)

        # Default models per provider
        if not config.model:
            config.model = _default_model(config.provider)

        return config


def _auto_detect_provider() -> str:
    """Auto-detect the best available provider."""
    if os.environ.get("DEEPSEEK_API_KEY"):
        return "deepseek"  # Cheapest
    if os.environ.get("OPENAI_API_KEY"):
        return "openai"
    if os.environ.get("ANTHROPIC_API_KEY"):
        return "anthropic"
    return "openai"


def _resolve_credentials(provider: str) -> Tuple[str, str]:
    """Resolve API key and base URL for a provider."""
    if provider == "deepseek":
        return (
            os.environ.get("DEEPSEEK_API_KEY", ""),
            os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        )
    if provider == "anthropic":
        return (
            os.environ.get("ANTHROPIC_API_KEY", ""),
            os.environ.get("ANTHROPIC_BASE_URL", "https://api.anthropic.com"),
        )
    if provider == "openai":
        return (
            os.environ.get("OPENAI_API_KEY", ""),
            os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
        )
    # OpenRouter fallback
    if os.environ.get("OPENROUTER_API_KEY"):
        return (
            os.environ.get("OPENROUTER_API_KEY", ""),
            "https://openrouter.ai/api/v1",
        )
    return ("", "")


def _default_model(provider: str) -> str:
    """Default model for auxiliary tasks (cheap + fast)."""
    defaults = {
        "deepseek": "deepseek-chat",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-haiku-20241022",
    }
    return defaults.get(provider, "gpt-4o-mini")


# ── Main API ──

def call_llm(
    prompt: str,
    system: str = "",
    config: Optional[AuxiliaryConfig] = None,
    task: str = "",
) -> str:
    """Call the auxiliary LLM with a simple prompt.

    This is the main entry point for auxiliary tasks like:
    - Context summarization
    - Message classification
    - Content analysis
    - Translation
    """
    config = config or AuxiliaryConfig.from_env(task)

    if not config.api_key:
        logger.warning("No auxiliary LLM configured (no API key)")
        return ""

    messages: List[Dict[str, Any]] = []
    if system:
        messages.append({"role": "system", "content": system})
    messages.append({"role": "user", "content": prompt})

    if config.provider == "anthropic":
        return _call_anthropic(messages, config)
    else:
        return _call_openai_compatible(messages, config)


def _call_openai_compatible(
    messages: List[Dict[str, Any]], config: AuxiliaryConfig,
) -> str:
    """Call OpenAI-compatible API."""
    import urllib.request

    payload = json.dumps({
        "model": config.model,
        "messages": messages,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }).encode()

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {config.api_key}",
    }

    try:
        req = urllib.request.Request(
            f"{config.base_url}/chat/completions",
            data=payload, headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read())
        return data.get("choices", [{}])[0].get("message", {}).get("content", "")
    except Exception as e:
        logger.warning("Auxiliary LLM call failed (%s): %s", config.provider, e)
        return ""


def _call_anthropic(
    messages: List[Dict[str, Any]], config: AuxiliaryConfig,
) -> str:
    """Call Anthropic API."""
    import urllib.request

    system = ""
    non_system = []
    for msg in messages:
        if msg.get("role") == "system":
            system = msg.get("content", "")
        else:
            non_system.append(msg)

    payload: Dict[str, Any] = {
        "model": config.model,
        "messages": non_system,
        "max_tokens": config.max_tokens,
        "temperature": config.temperature,
    }
    if system:
        payload["system"] = system

    headers = {
        "Content-Type": "application/json",
        "x-api-key": config.api_key,
        "anthropic-version": "2023-06-01",
    }

    try:
        req = urllib.request.Request(
            f"{config.base_url}/v1/messages",
            data=json.dumps(payload).encode(),
            headers=headers, method="POST",
        )
        with urllib.request.urlopen(req, timeout=config.timeout) as resp:
            data = json.loads(resp.read())
        text = ""
        for block in data.get("content", []):
            if block.get("type") == "text":
                text += block.get("text", "")
        return text
    except Exception as e:
        logger.warning("Auxiliary LLM call failed (anthropic): %s", e)
        return ""


# ── Convenience Functions ──

def summarize(text: str, max_words: int = 200) -> str:
    """Summarize text using auxiliary LLM."""
    return call_llm(
        f"Summarize the following in {max_words} words or less:\n\n{text}",
        system="You are a concise summarizer. Output only the summary.",
        task="summarize",
    )


def classify(text: str, categories: List[str]) -> str:
    """Classify text into one of the given categories."""
    cats = ", ".join(categories)
    return call_llm(
        f"Classify the following text into exactly one category.\n"
        f"Categories: {cats}\n\nText: {text}\n\nCategory:",
        system="Output only the category name, nothing else.",
        task="classify",
    )


def translate(text: str, target_language: str) -> str:
    """Translate text to target language."""
    return call_llm(
        f"Translate to {target_language}:\n\n{text}",
        system=f"You are a translator. Output only the {target_language} translation.",
        task="translate",
    )


# ── Fallback Chain (for agent loop) ──

class _FallbackChain:
    """Provides fallback LLM provider when primary fails."""

    def __init__(self, config: AuxiliaryConfig):
        self._config = config
        self._activated = False

    @property
    def has_fallbacks(self) -> bool:
        return bool(self._config.provider) and not self._activated

    def try_activate_next(self):
        """Try to create a fallback provider. Returns provider or None."""
        if self._activated:
            return None
        self._activated = True
        try:
            from caveman.providers.openai_adapter import OpenAIProvider
            return OpenAIProvider(
                api_key=self._config.api_key,
                model=self._config.model,
                base_url=self._config.base_url,
            )
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return None
