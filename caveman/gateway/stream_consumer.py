"""Gateway stream consumer — bridges sync agent callbacks to async platform delivery.

The agent fires stream_delta_callback(text) synchronously from its worker thread.
StreamConsumer:
  1. Receives deltas via on_delta() (thread-safe, sync)
  2. Queues them to an asyncio task via queue.Queue
  3. The async run() task buffers, rate-limits, and progressively edits
     a single message on the target platform

Uses the edit transport (send initial message, then edit) — works across
Discord, Telegram, and Slack.
"""
from __future__ import annotations

import asyncio
import logging
import queue
import time
from dataclasses import dataclass
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

_DONE = object()
_NEW_SEGMENT = object()

# Rate limit: minimum interval between edits (seconds)
_MIN_EDIT_INTERVAL = 0.8
_MIN_CHARS_PER_EDIT = 20
_MAX_MESSAGE_LENGTH = 4000


@dataclass
class StreamConfig:
    """Configuration for stream consumer."""
    min_edit_interval: float = _MIN_EDIT_INTERVAL
    min_chars_per_edit: int = _MIN_CHARS_PER_EDIT
    max_message_length: int = _MAX_MESSAGE_LENGTH
    typing_indicator: bool = True


class StreamConsumer:
    """Bridges sync agent stream deltas to async platform message edits."""

    def __init__(
        self,
        send_fn: Callable[[str], Awaitable[str]],
        edit_fn: Callable[[str, str], Awaitable[None]],
        config: StreamConfig | None = None,
    ) -> None:
        """
        Args:
            send_fn: async fn(text) -> message_id. Sends initial message.
            edit_fn: async fn(message_id, text) -> None. Edits existing message.
            config: Stream configuration.
        """
        self._send = send_fn
        self._edit = edit_fn
        self._config = config or StreamConfig()
        self._queue: queue.Queue = queue.Queue()
        self._buffer = ""
        self._message_id: str = ""
        self._last_edit_time = 0.0
        self._done = False

    def on_delta(self, text: str) -> None:
        """Receive a stream delta (thread-safe, sync)."""
        self._queue.put(text)

    def on_tool_boundary(self) -> None:
        """Signal a tool boundary — finalize current message, start new one."""
        self._queue.put(_NEW_SEGMENT)

    def on_complete(self) -> None:
        """Signal stream completion."""
        self._queue.put(_DONE)

    async def run(self) -> str:
        """Consume the stream and deliver to platform.

        Returns the final accumulated text.
        """
        while not self._done:
            try:
                item = self._queue.get(timeout=0.1)
            except queue.Empty:
                await asyncio.sleep(0.05)
                continue

            if item is _DONE:
                self._done = True
                break

            if item is _NEW_SEGMENT:
                # Finalize current message
                if self._buffer and self._message_id:
                    await self._flush()
                self._message_id = ""
                self._buffer = ""
                continue

            # Accumulate text
            self._buffer += item
            await self._maybe_flush()

        # Final flush
        if self._buffer:
            await self._flush()

        return self._buffer

    async def _maybe_flush(self) -> None:
        """Flush if enough time/chars have accumulated."""
        now = time.time()
        elapsed = now - self._last_edit_time
        if elapsed < self._config.min_edit_interval:
            return
        if len(self._buffer) < self._config.min_chars_per_edit and not self._done:
            return
        await self._flush()

    async def _flush(self) -> None:
        """Send or edit the current buffer."""
        if not self._buffer:
            return

        text = self._buffer
        if len(text) > self._config.max_message_length:
            text = text[:self._config.max_message_length] + "..."

        try:
            if not self._message_id:
                self._message_id = await self._send(text)
            else:
                await self._edit(self._message_id, text)
            self._last_edit_time = time.time()
        except Exception as e:
            logger.warning("Stream flush error: %s", e)
