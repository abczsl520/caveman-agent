"""Model fallback chain — automatic failover when primary model fails.

Ported from Hermes _try_activate_fallback (MIT, Nous Research).

Config example:
  agent:
    fallback_chain:
      - provider: anthropic
        model: claude-sonnet-4-20250514
      - provider: openai
        model: gpt-4o
"""
from __future__ import annotations
import logging
from dataclasses import dataclass

from caveman.providers.llm import LLMProvider

logger = logging.getLogger(__name__)


@dataclass
class FallbackEntry:
    """A provider entry in the fallback chain with priority and health state."""
    provider: str
    model: str
    api_key: str = ""
    base_url: str = ""


class FallbackChain:
    """Manages a chain of fallback models for automatic failover."""

    def __init__(self, entries: list[dict[str, str]] | None = None):
        self._entries: list[FallbackEntry] = []
        self._index = 0
        self._original_provider: LLMProvider | None = None
        self._active_provider: LLMProvider | None = None
        if entries:
            for e in entries:
                self._entries.append(FallbackEntry(
                    provider=e.get("provider", ""),
                    model=e.get("model", ""),
                    api_key=e.get("api_key", ""),
                    base_url=e.get("base_url", ""),
                ))

    @property
    def has_fallbacks(self) -> bool:
        return self._index < len(self._entries)

    @property
    def exhausted(self) -> bool:
        return self._index >= len(self._entries)

    def try_activate_next(self) -> LLMProvider | None:
        """Try to activate the next fallback. Returns new provider or None."""
        while self._index < len(self._entries):
            entry = self._entries[self._index]
            self._index += 1
            if not entry.provider or not entry.model:
                continue
            try:
                provider = self._create_provider(entry)
                if provider:
                    logger.warning(
                        "Fallback activated: %s/%s (index %d/%d)",
                        entry.provider, entry.model,
                        self._index, len(self._entries),
                    )
                    self._active_provider = provider
                    return provider
            except Exception as e:
                logger.warning("Fallback %s/%s failed to init: %s",
                               entry.provider, entry.model, e)
        logger.error("All fallbacks exhausted")
        return None

    def _create_provider(self, entry: FallbackEntry) -> LLMProvider | None:
        """Create a provider instance from a fallback entry."""
        import os
        if entry.provider == "anthropic":
            from caveman.providers.anthropic_provider import AnthropicProvider
            key = entry.api_key or os.environ.get("ANTHROPIC_API_KEY", "")
            return AnthropicProvider(api_key=key, model=entry.model)
        elif entry.provider in ("openai", "deepseek"):
            from caveman.providers.openai_provider import OpenAIProvider
            if entry.provider == "deepseek":
                key = entry.api_key or os.environ.get("DEEPSEEK_API_KEY", "")
                base = entry.base_url or "https://api.deepseek.com/v1"
            else:
                key = entry.api_key or os.environ.get("OPENAI_API_KEY", "")
                base = entry.base_url or None
            return OpenAIProvider(api_key=key, model=entry.model, base_url=base)
        elif entry.provider == "ollama":
            from caveman.providers.ollama_provider import OllamaProvider
            base = entry.base_url or "http://localhost:11434"
            return OllamaProvider(model=entry.model, base_url=base)
        logger.warning("Unknown fallback provider: %s", entry.provider)
        return None

    def reset(self) -> None:
        """Reset the chain (e.g. on new session)."""
        self._index = 0
        self._active_provider = None
