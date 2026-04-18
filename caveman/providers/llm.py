"""LLM provider abstraction — unified interface across providers.

All providers yield normalized event dicts:
- {"type": "delta", "text": "..."}
- {"type": "tool_call", "id": "...", "name": "...", "input": {...}}
- {"type": "done", "stop_reason": "end_turn"|"max_tokens"|"tool_use"}

Architecture:
- complete() is the ONLY public API — callers iterate events
- _build_params() normalizes input → provider-specific API params
- _stream() / _non_stream() are internal implementation details
- Retry is scoped to the API call, NOT the stream consumption
"""
from __future__ import annotations
import logging
from abc import ABC, abstractmethod
from typing import AsyncIterator, Any

logger = logging.getLogger(__name__)


# Canonical stop_reason values used throughout Caveman
STOP_END_TURN = "end_turn"
STOP_MAX_TOKENS = "max_tokens"
STOP_TOOL_USE = "tool_use"

# Normalization map: provider-specific → canonical
_STOP_REASON_MAP = {
    # Anthropic (already canonical)
    "end_turn": STOP_END_TURN,
    "max_tokens": STOP_MAX_TOKENS,
    "tool_use": STOP_TOOL_USE,
    "stop_sequence": STOP_END_TURN,
    # OpenAI
    "stop": STOP_END_TURN,
    "length": STOP_MAX_TOKENS,
    "tool_calls": STOP_TOOL_USE,
    "content_filter": STOP_END_TURN,
    "function_call": STOP_TOOL_USE,
    # Gemini (pre-normalized, but just in case)
    "STOP": STOP_END_TURN,
    "MAX_TOKENS": STOP_MAX_TOKENS,
    "SAFETY": STOP_END_TURN,
    # Ollama
    "complete": STOP_END_TURN,
}


def normalize_stop_reason(raw: str | None) -> str:
    """Normalize any provider's stop_reason to canonical Caveman values.

    Returns STOP_END_TURN for unknown/empty values (safe default: terminate).
    """
    if not raw:
        return STOP_END_TURN
    return _STOP_REASON_MAP.get(raw, STOP_END_TURN)


class LLMProvider(ABC):
    """Abstract LLM provider. All providers yield normalized events.

    Subclasses must implement:
    - complete() — the main generation method (yields normalized events)
    - context_length — the model's context window
    - _get_client() — return a configured API client
    - _build_params() — convert (messages, system, tools) → provider-specific params

    Common behavior (shared):
    - model, max_tokens attributes
    - model_info property
    """

    model: str
    max_tokens: int

    @abstractmethod
    async def complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        system: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Generate completion. Yields: delta/tool_call/done events."""
        ...

    async def safe_complete(
        self,
        messages: list[dict],
        tools: list[dict] | None = None,
        stream: bool = True,
        system: str | None = None,
        **kwargs,
    ) -> AsyncIterator[dict]:
        """Wrapper around complete() with system prompt guardrail.

        Use this instead of complete() to catch empty system prompts
        regardless of which provider is active.
        Internal LLM calls (nudge/reflect/shield) pass no system prompt — that's OK.
        """
        # Only warn if tools are provided (= agent task, not internal LLM call)
        if tools and (not system or len(system) < 50):
            logger.warning("⚠️ System prompt empty or short (%d chars) — likely a bug",
                           len(system) if system else 0)
        async for event in self.complete(messages, tools, stream, system, **kwargs):
            yield event

    @property
    @abstractmethod
    def context_length(self) -> int:
        """Maximum context window in tokens."""
        ...

    @abstractmethod
    def _get_client(self) -> Any:
        """Return configured API client (lazy-initialized)."""
        ...

    @abstractmethod
    def _build_params(self, messages: list[dict], system: str | None = None, tools: list[dict] | None = None, **kwargs) -> dict[str, Any]:
        """Convert common args to provider-specific API params."""
        ...

    @property
    def model_info(self) -> dict[str, Any]:
        """Return provider metadata for logging/events."""
        return {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "context_length": self.context_length,
            "provider": type(self).__name__,
        }
