"""Dispatch Depth — streaming, TTS integration, block reply, tool status.

Supplements dispatch.py with streaming output, TTS auto-apply, and
block-based reply chunking. Extracted from OpenClaw dispatch-from-config.ts.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

__all__ = [
    "StreamChunk",
    "BlockReplyConfig",
    "TTSConfig",
    "StreamingDispatcher",
]


logger = logging.getLogger("caveman.gateway.dispatch_depth")


@dataclass
class StreamChunk:
    """A chunk of streaming output."""
    text: str = ""
    is_final: bool = False
    tool_name: str = ""
    tool_status: str = ""  # start | progress | complete | error
    tool_result_preview: str = ""
    phase: str = ""  # thinking | writing | tool_call


@dataclass
class BlockReplyConfig:
    """Configuration for block-based reply chunking."""
    min_chars: int = 200
    max_chars: int = 2000
    break_preference: str = "paragraph"  # paragraph | newline | sentence
    flush_on_paragraph: bool = True


@dataclass
class TTSConfig:
    """TTS auto-apply configuration."""
    enabled: bool = False
    provider: str = "system"
    voice: str = ""
    max_chars: int = 4000
    summary_mode: bool = False  # Summarize long text before TTS


class StreamingDispatcher:
    """Handles streaming output with block-based chunking."""

    def __init__(
        self,
        send_fn: Optional[Callable] = None,
        edit_fn: Optional[Callable] = None,
        typing_fn: Optional[Callable] = None,
        tts_config: Optional[TTSConfig] = None,
        block_config: Optional[BlockReplyConfig] = None,
    ):
        self._send_fn = send_fn
        self._edit_fn = edit_fn
        self._typing_fn = typing_fn
        self._tts = tts_config or TTSConfig()
        self._block = block_config or BlockReplyConfig()
        self._buffer = ""
        self._sent_message_id: str = ""
        self._chunks_sent = 0
        self._tool_statuses: Dict[str, str] = {}

    async def handle_chunk(self, chunk: StreamChunk) -> None:
        """Process a streaming chunk."""
        if chunk.tool_name:
            await self._handle_tool_status(chunk)
            return

        self._buffer += chunk.text

        if chunk.is_final:
            await self._flush_final()
        elif self._should_flush():
            await self._flush_block()

    async def _handle_tool_status(self, chunk: StreamChunk) -> None:
        """Handle tool execution status updates."""
        self._tool_statuses[chunk.tool_name] = chunk.tool_status

        if chunk.tool_status == "start" and self._send_fn:
            # Send typing or status indicator
            if self._typing_fn:
                try:
                    result = self._typing_fn()
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.debug("_handle_tool_status: suppressed %s", exc)

    def _should_flush(self) -> bool:
        """Check if buffer should be flushed."""
        if len(self._buffer) < self._block.min_chars:
            return False
        if len(self._buffer) >= self._block.max_chars:
            return True

        # Check for natural break points
        if self._block.break_preference == "paragraph":
            return "\n\n" in self._buffer[-100:]
        elif self._block.break_preference == "newline":
            return "\n" in self._buffer[-50:]
        elif self._block.break_preference == "sentence":
            return bool(re.search(r'[.!?]\s', self._buffer[-50:]))

        return False

    async def _flush_block(self) -> None:
        """Flush a block of text."""
        text = self._find_break_point()
        if not text:
            return

        if self._edit_fn and self._sent_message_id:
            # Edit existing message (streaming update)
            try:
                result = self._edit_fn(self._sent_message_id, text)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug("_flush_block: suppressed %s", exc)
        elif self._send_fn:
            try:
                result = self._send_fn(text)
                if hasattr(result, "__await__"):
                    result = await result
                if isinstance(result, dict):
                    self._sent_message_id = result.get("message_id", "")
            except Exception as exc:
                logger.debug("_flush_block: suppressed %s", exc)

        self._chunks_sent += 1

    async def _flush_final(self) -> None:
        """Flush remaining buffer as final message."""
        if not self._buffer.strip():
            return

        text = self._buffer.strip()
        self._buffer = ""

        if self._edit_fn and self._sent_message_id:
            try:
                result = self._edit_fn(self._sent_message_id, text)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug("_flush_final: suppressed %s", exc)
        elif self._send_fn:
            try:
                result = self._send_fn(text)
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug("_flush_final: suppressed %s", exc)

        # Auto-TTS
        if self._tts.enabled:
            await self._apply_tts(text)

    async def _apply_tts(self, text: str) -> None:
        """Apply TTS to the final response."""
        if len(text) > self._tts.max_chars:
            if self._tts.summary_mode:
                # Would need LLM to summarize — placeholder
                text = text[:self._tts.max_chars]
            else:
                text = text[:self._tts.max_chars]

        try:
            from caveman.tools.builtin.tts_v2 import tts_generate
            await tts_generate(text, self._tts.provider, self._tts.voice)
        except Exception as e:
            logger.debug("TTS failed: %s", e)

    def _find_break_point(self) -> str:
        """Find a natural break point in the buffer."""
        max_len = self._block.max_chars

        if len(self._buffer) <= max_len:
            text = self._buffer
            self._buffer = ""
            return text

        # Look for paragraph break
        if self._block.break_preference == "paragraph":
            idx = self._buffer.rfind("\n\n", 0, max_len)
            if idx > self._block.min_chars:
                text = self._buffer[:idx]
                self._buffer = self._buffer[idx:].lstrip("\n")
                return text

        # Look for newline break
        idx = self._buffer.rfind("\n", 0, max_len)
        if idx > self._block.min_chars:
            text = self._buffer[:idx]
            self._buffer = self._buffer[idx:].lstrip("\n")
            return text

        # Hard break at max
        text = self._buffer[:max_len]
        self._buffer = self._buffer[max_len:]
        return text

    def get_stats(self) -> Dict[str, Any]:
        return {
            "chunks_sent": self._chunks_sent,
            "buffer_size": len(self._buffer),
            "tool_statuses": dict(self._tool_statuses),
        }
