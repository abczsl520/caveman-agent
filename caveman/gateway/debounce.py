"""Inbound message debounce — merge rapid-fire messages into one task.

When a user sends multiple messages quickly (common in Discord/Telegram),
this module collects them and delivers a single merged task.
"""
from __future__ import annotations

import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Callable, Awaitable

logger = logging.getLogger(__name__)

DEFAULT_DEBOUNCE_MS = 1500  # Wait 1.5s after last message before processing


@dataclass
class _PendingBatch:
    messages: list[str] = field(default_factory=list)
    contexts: list[dict] = field(default_factory=list)
    timer: asyncio.TimerHandle | None = None
    future: asyncio.Future | None = None


class MessageDebouncer:
    """Collects rapid-fire messages from the same user/channel and merges them."""

    def __init__(
        self,
        handler: Callable[[str, dict], Awaitable[None]],
        debounce_ms: int = DEFAULT_DEBOUNCE_MS,
    ):
        self._handler = handler
        self._debounce_s = debounce_ms / 1000.0
        self._pending: dict[str, _PendingBatch] = defaultdict(_PendingBatch)
        self._loop: asyncio.AbstractEventLoop | None = None

    def _get_key(self, context: dict) -> str:
        """Generate a unique key for user+channel combination."""
        gw = context.get("gateway_name", "")
        ch = context.get("channel_id", "")
        user = context.get("user_id", "")
        return f"{gw}:{ch}:{user}"

    async def add_message(self, text: str, context: dict) -> None:
        """Add a message to the debounce queue."""
        if self._loop is None:
            self._loop = asyncio.get_running_loop()

        key = self._get_key(context)
        batch = self._pending[key]

        # Cancel existing timer
        if batch.timer:
            batch.timer.cancel()

        batch.messages.append(text)
        batch.contexts.append(context)

        # Set new timer
        batch.timer = self._loop.call_later(
            self._debounce_s,
            lambda k=key: asyncio.create_task(self._flush(k)),
        )

    async def _flush(self, key: str) -> None:
        """Flush a pending batch — merge messages and call handler."""
        batch = self._pending.pop(key, None)
        if not batch or not batch.messages:
            return

        # Merge messages
        if len(batch.messages) == 1:
            merged = batch.messages[0]
        else:
            merged = "\n".join(batch.messages)
            logger.info("Debounced %d messages from %s", len(batch.messages), key)

        # Use the last context (most recent message)
        context = batch.contexts[-1]

        try:
            await self._handler(merged, context)
        except Exception as e:
            logger.error("Debounced handler error for %s: %s", key, e)

    async def flush_all(self) -> None:
        """Flush all pending batches (for shutdown)."""
        keys = list(self._pending.keys())
        for key in keys:
            batch = self._pending.get(key)
            if batch and batch.timer:
                batch.timer.cancel()
            await self._flush(key)
