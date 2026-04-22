"""Outbound Message Delivery — chunk, format, and deliver responses.

Extracted from OpenClaw send.outbound.ts (601 lines) and
Hermes BasePlatformAdapter._send_with_retry + truncate_message.

Features:
- Code-block-aware chunking (never break mid-fence)
- Platform-specific formatting (Discord markdown, Telegram HTML)
- Retry with exponential backoff
- Format degradation (markdown → plain text on failure)
- Reaction management (ack → processing → done)
- Message editing for streaming updates
"""
from __future__ import annotations

import asyncio
import logging
import re
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from caveman.gateway.platform_types import SendResult

__all__ = [
    "PLATFORM_LIMITS",
    "MAX_RETRIES",
    "RETRY_BASE_DELAY",
    "RETRY_BACKOFF",
    "RETRYABLE_PATTERNS",
    "ReactionState",
    "OutboundDelivery",
    "chunk_message",
    "strip_markdown",
]


logger = logging.getLogger("caveman.gateway.outbound")

# Platform limits
PLATFORM_LIMITS = {
    "discord": 2000,
    "telegram": 4096,
    "slack": 3000,
    "whatsapp": 4096,
    "signal": 6000,
    "matrix": 16384,
    "feishu": 4096,
    "default": 4096,
}

# Retry config
MAX_RETRIES = 3
RETRY_BASE_DELAY = 1.0
RETRY_BACKOFF = 2.0
RETRYABLE_PATTERNS = frozenset({
    "rate limit", "429", "503", "502", "timeout",
    "connection", "temporarily", "overloaded",
})


@dataclass
class ReactionState:
    """Track reaction lifecycle for a message."""
    channel_id: str
    message_id: str
    ack_emoji: str = "👀"
    processing_emoji: str = "⏳"
    done_emoji: str = "✅"
    error_emoji: str = "❌"
    current: str = ""


class OutboundDelivery:
    """Handles all outbound message delivery for a platform adapter."""

    def __init__(self, adapter: Any, platform: str = "default"):
        self._adapter = adapter
        self._platform = platform
        self._max_length = PLATFORM_LIMITS.get(platform, PLATFORM_LIMITS["default"])

    # ── Main Send ──

    async def send_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SendResult:
        """Send message with chunking, retry, and format degradation."""
        if not content.strip():
            return SendResult(success=True)

        chunks = chunk_message(content, self._max_length)
        last_result = SendResult(success=False, error="no chunks")

        for i, chunk in enumerate(chunks):
            # Only reply_to on first chunk
            rt = reply_to if i == 0 else None
            last_result = await self._send_single_with_retry(chat_id, chunk, rt, metadata)
            if not last_result.success:
                # Try format degradation
                plain = strip_markdown(chunk)
                if plain != chunk:
                    logger.debug("Retrying with plain text after markdown failure")
                    last_result = await self._send_single_with_retry(
                        chat_id, plain, rt, metadata)
                if not last_result.success:
                    break

        return last_result

    async def _send_single_with_retry(
        self,
        chat_id: str,
        content: str,
        reply_to: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> SendResult:
        """Send a single chunk with retry and backoff."""
        delay = RETRY_BASE_DELAY

        for attempt in range(MAX_RETRIES + 1):
            try:
                result = await self._adapter.send(
                    chat_id, content, reply_to=reply_to, metadata=metadata)
                if result.success:
                    return result
                if not _is_retryable(result.error or ""):
                    return result
            except Exception as e:
                if attempt == MAX_RETRIES:
                    return SendResult(success=False, error=str(e))
                if not _is_retryable(str(e)):
                    return SendResult(success=False, error=str(e))

            if attempt < MAX_RETRIES:
                logger.debug("Retry %d/%d after %.1fs", attempt + 1, MAX_RETRIES, delay)
                await asyncio.sleep(delay)
                delay *= RETRY_BACKOFF

        return SendResult(success=False, error="max retries exceeded")

    # ── Streaming Edit ──

    async def edit_streaming(
        self,
        chat_id: str,
        message_id: str,
        content: str,
        metadata: Optional[Dict] = None,
    ) -> SendResult:
        """Edit a message for streaming updates."""
        if hasattr(self._adapter, "edit_message"):
            try:
                return await self._adapter.edit_message(
                    chat_id, message_id, content, metadata=metadata)
            except Exception as e:
                return SendResult(success=False, error=str(e))
        return SendResult(success=False, error="edit not supported")

    # ── Reactions ──

    async def add_reaction(
        self, chat_id: str, message_id: str, emoji: str,
    ) -> bool:
        """Add a reaction to a message."""
        if hasattr(self._adapter, "add_reaction"):
            try:
                await self._adapter.add_reaction(chat_id, message_id, emoji)
                return True
            except Exception as exc:
                logger.debug("add_reaction: suppressed %s", exc)
        return False

    async def remove_reaction(
        self, chat_id: str, message_id: str, emoji: str,
    ) -> bool:
        """Remove a reaction from a message."""
        if hasattr(self._adapter, "remove_reaction"):
            try:
                await self._adapter.remove_reaction(chat_id, message_id, emoji)
                return True
            except Exception:
                pass  # intentional: Exception suppressed
        return False

    async def set_processing_reaction(self, state: ReactionState) -> None:
        """Transition reaction: ack → processing."""
        await self.remove_reaction(state.channel_id, state.message_id, state.ack_emoji)
        await self.add_reaction(state.channel_id, state.message_id, state.processing_emoji)
        state.current = state.processing_emoji

    async def set_done_reaction(self, state: ReactionState, success: bool = True) -> None:
        """Transition reaction: processing → done/error."""
        if state.current:
            await self.remove_reaction(state.channel_id, state.message_id, state.current)
        emoji = state.done_emoji if success else state.error_emoji
        await self.add_reaction(state.channel_id, state.message_id, emoji)
        state.current = emoji


# ── Chunking ──

def chunk_message(content: str, max_length: int = 4096) -> List[str]:
    """Split message into chunks, preserving code blocks.

    Delegates to caveman.utils.split_message (code-fence-aware).
    """
    from caveman.utils import split_message
    return split_message(content, max_length=max_length)


def strip_markdown(text: str) -> str:
    """Strip markdown formatting for plain text fallback."""
    # Remove code blocks
    text = re.sub(r"```[\s\S]*?```", lambda m: m.group(0).strip("`"), text)
    # Remove inline code
    text = re.sub(r"`([^`]+)`", r"\1", text)
    # Remove bold/italic
    text = re.sub(r"\*\*(.+?)\*\*", r"\1", text)
    text = re.sub(r"\*(.+?)\*", r"\1", text)
    text = re.sub(r"__(.+?)__", r"\1", text)
    text = re.sub(r"_(.+?)_", r"\1", text)
    # Remove headers
    text = re.sub(r"^#{1,6}\s+", "", text, flags=re.MULTILINE)
    # Remove links but keep text
    text = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", text)
    return text


def _is_retryable(error: str) -> bool:
    """Check if an error is worth retrying."""
    lower = error.lower()
    return any(pat in lower for pat in RETRYABLE_PATTERNS)
