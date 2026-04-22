"""Processor Depth — streaming output, tool progress, retry logic.

Supplements processor.py with streaming support and tool execution
progress tracking. Extracted from OpenClaw gateway processor.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "ToolProgressEvent",
    "RetryConfig",
    "StreamingProcessor",
]


logger = logging.getLogger("caveman.gateway.processor_depth")


@dataclass
class ToolProgressEvent:
    """Progress event for a tool execution."""
    tool_name: str
    status: str  # queued | running | completed | failed | cancelled
    started_at: float = 0
    progress_pct: float = 0
    message: str = ""
    result_preview: str = ""


@dataclass
class RetryConfig:
    """Configuration for automatic retry."""
    max_retries: int = 3
    base_delay: float = 1.0
    max_delay: float = 30.0
    backoff_factor: float = 2.0
    retryable_errors: tuple = (
        "rate_limit", "timeout", "server_error", "overloaded",
    )

    def delay_for(self, attempt: int) -> float:
        delay = self.base_delay * (self.backoff_factor ** attempt)
        return min(delay, self.max_delay)


class StreamingProcessor:
    """Message processor with streaming output and retry support."""

    def __init__(
        self,
        process_fn: Optional[Callable] = None,
        retry_config: Optional[RetryConfig] = None,
        on_tool_progress: Optional[Callable] = None,
        on_stream_chunk: Optional[Callable] = None,
    ):
        self._process_fn = process_fn
        self._retry = retry_config or RetryConfig()
        self._on_tool_progress = on_tool_progress
        self._on_stream_chunk = on_stream_chunk
        self._tool_events: List[ToolProgressEvent] = []

    async def process_with_retry(
        self,
        message: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Process a message with automatic retry on transient errors."""
        last_error = None
        for attempt in range(self._retry.max_retries + 1):
            try:
                result = await self._do_process(message, context)
                return result
            except Exception as e:
                error_type = self._classify_error(e)
                if error_type not in self._retry.retryable_errors:
                    raise
                last_error = e
                if attempt < self._retry.max_retries:
                    delay = self._retry.delay_for(attempt)
                    logger.info(
                        "Retrying (%d/%d) after %.1fs: %s",
                        attempt + 1, self._retry.max_retries, delay, error_type,
                    )
                    await asyncio.sleep(delay)

        raise last_error or RuntimeError("All retries exhausted")

    async def _do_process(
        self, message: str, context: Optional[Dict[str, Any]],
    ) -> Dict[str, Any]:
        if not self._process_fn:
            return {"text": "", "error": "No process function configured"}
        result = self._process_fn(message, context)
        if hasattr(result, "__await__"):
            result = await result
        return result if isinstance(result, dict) else {"text": str(result)}

    def _classify_error(self, error: Exception) -> str:
        msg = str(error).lower()
        if "rate" in msg or "429" in msg:
            return "rate_limit"
        if "timeout" in msg or "timed out" in msg:
            return "timeout"
        if "500" in msg or "502" in msg or "503" in msg:
            return "server_error"
        if "overloaded" in msg or "capacity" in msg:
            return "overloaded"
        return "unknown"

    def report_tool_progress(self, event: ToolProgressEvent) -> None:
        """Report tool execution progress."""
        self._tool_events.append(event)
        if self._on_tool_progress:
            try:
                result = self._on_tool_progress(event)
                if hasattr(result, "__await__"):
                    asyncio.ensure_future(result)
            except Exception as exc:
                logger.debug("report_tool_progress: suppressed %s", exc)

    def get_tool_summary(self) -> List[Dict[str, Any]]:
        return [
            {
                "tool": e.tool_name,
                "status": e.status,
                "duration_ms": round((time.monotonic() - e.started_at) * 1000, 1) if e.started_at else 0,
            }
            for e in self._tool_events
        ]
