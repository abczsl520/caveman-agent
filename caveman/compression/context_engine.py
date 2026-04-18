"""Context Engine — pluggable interface for context management.

Ported from Hermes agent/context_engine.py (184 lines).
Defines the ABC that all context engines must implement.
The built-in SmartCompressor is the default implementation.
"""
from __future__ import annotations

__all__ = ["ContextEngine"]

import logging
from abc import ABC, abstractmethod
from typing import Any

logger = logging.getLogger(__name__)


class ContextEngine(ABC):
    """Base class for pluggable context management engines.

    A context engine controls how conversation context is managed when
    approaching the model's token limit. Responsibilities:
      - Deciding when compaction should fire
      - Performing compaction (summarization, etc.)
      - Tracking token usage from API responses

    Usage:
        class MyEngine(ContextEngine):
            @property
            def name(self) -> str:
                return "my-engine"
            ...

        engine = MyEngine(context_length=200_000)
        engine.update_from_response({"usage": {"input_tokens": 5000}})
        if engine.should_compress():
            result = await engine.compress(messages)
    """

    # Token state — engines MUST maintain these
    last_prompt_tokens: int = 0
    last_completion_tokens: int = 0
    last_total_tokens: int = 0
    threshold_tokens: int = 0
    context_length: int = 0
    compression_count: int = 0

    # Compaction parameters
    threshold_percent: float = 0.75
    protect_first_n: int = 3

    def __init__(
        self,
        context_length: int = 128_000,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
    ) -> None:
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.threshold_tokens = int(context_length * threshold_percent)
        self.protect_first_n = protect_first_n

    @property
    @abstractmethod
    def name(self) -> str:
        """Short identifier (e.g. 'compressor', 'lcm')."""

    @abstractmethod
    def update_from_response(self, response: dict[str, Any]) -> None:
        """Update token counts from an API response."""

    @abstractmethod
    def should_compress(self) -> bool:
        """Return True if compaction should run now."""

    @abstractmethod
    async def compress(
        self,
        messages: list[dict[str, Any]],
        **kwargs: Any,
    ) -> list[dict[str, Any]]:
        """Compress messages and return the new message list."""

    def on_session_start(self, session_id: str = "") -> None:
        """Called when a conversation session begins."""
        self.last_prompt_tokens = 0
        self.last_completion_tokens = 0
        self.last_total_tokens = 0
        self.compression_count = 0

    def on_session_end(self, session_id: str = "") -> None:
        """Called at real session boundaries."""

    @property
    def utilization(self) -> float:
        """Current context utilization (0.0 to 1.0+)."""
        if self.context_length <= 0:
            return 0.0
        return self.last_total_tokens / self.context_length

    @property
    def tokens_remaining(self) -> int:
        """Estimated tokens remaining before threshold."""
        return max(0, self.threshold_tokens - self.last_total_tokens)
