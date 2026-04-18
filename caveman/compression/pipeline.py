"""3-layer context compression pipeline.

Layer 1 (Micro): Dedup + whitespace normalization — always runs, ~10% reduction
Layer 2 (Normal): Tool result truncation + old message pruning — runs when >60% context
Layer 3 (Smart): Structured LLM summarization — runs when >80% context

Layer 3 is ported from Hermes ContextCompressor (MIT, Nous Research) with
OpenClaw identifier preservation (MIT, Peter Steinberger).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import Any

from caveman.compression.smart import SmartCompressor, estimate_tokens


@dataclass
class CompressionStats:
    """Track what compression did."""
    original_tokens: int = 0
    final_tokens: int = 0
    layer_applied: str = "none"
    messages_removed: int = 0
    messages_summarized: int = 0

    @property
    def ratio(self) -> float:
        if self.original_tokens == 0:
            return 1.0
        return self.final_tokens / self.original_tokens


class CompressionPipeline:
    """3-layer context compression."""

    def __init__(
        self,
        micro_threshold: float = 0.0,
        normal_threshold: float = 0.6,
        smart_threshold: float = 0.8,
        max_tool_result_len: int = 500,
        preserve_last_n: int = 10,
        provider: Any = None,
        context_length: int = 200_000,
    ):
        self.micro_threshold = micro_threshold
        self.normal_threshold = normal_threshold
        self.smart_threshold = smart_threshold
        self.max_tool_result_len = max_tool_result_len
        self.preserve_last_n = preserve_last_n
        self.provider = provider
        self._smart = SmartCompressor(
            provider=provider,
            context_length=context_length,
        )

    @property
    def smart_compressor(self) -> SmartCompressor:
        """Access the underlying SmartCompressor for Shield integration."""
        return self._smart

    async def compress(
        self, messages: list[dict], context_usage: float = 0.0,
        focus_topic: str | None = None,
    ) -> tuple[list[dict], CompressionStats]:
        """Run compression layers based on context usage ratio.

        Args:
            messages: Current conversation messages
            context_usage: 0.0-1.0 ratio of context window used
            focus_topic: Optional focus for guided compression

        Returns:
            (compressed_messages, stats)
        """
        stats = CompressionStats(original_tokens=estimate_tokens(messages))

        # Layer 1: Micro — always runs
        result = self._layer_micro(messages)
        stats.layer_applied = "micro"

        # Layer 2: Normal — when context >60%
        if context_usage >= self.normal_threshold:
            before = len(result)
            result = self._layer_normal(result)
            stats.messages_removed += before - len(result)
            stats.layer_applied = "normal"

        # Layer 3: Smart — when context >80%
        if context_usage >= self.smart_threshold:
            before_count = len(result)
            current_tokens = estimate_tokens(result)
            result = await self._smart.compress(
                result, current_tokens=current_tokens,
                focus_topic=focus_topic,
            )
            stats.messages_summarized = before_count - len(result)
            stats.layer_applied = "smart"

        stats.final_tokens = estimate_tokens(result)
        return result, stats

    def _layer_micro(self, messages: list[dict]) -> list[dict]:
        """Layer 1: Remove exact duplicates + normalize whitespace."""
        if len(messages) <= 2:
            return list(messages)

        result = [messages[0]]

        for msg in messages[1:]:
            h = _msg_hash(msg)
            if h == _msg_hash(result[-1]):
                continue
            if isinstance(msg.get("content"), str):
                msg = {**msg, "content": _normalize_whitespace(msg["content"])}
            result.append(msg)

        return result

    def _layer_normal(self, messages: list[dict]) -> list[dict]:
        """Layer 2: Truncate tool results + remove old low-value messages."""
        result = []
        for i, msg in enumerate(messages):
            content = msg.get("content", "")

            # Truncate long tool results
            if msg.get("role") == "tool" and isinstance(content, str) and len(content) > self.max_tool_result_len:
                head = content[:self.max_tool_result_len // 2]
                tail = content[-self.max_tool_result_len // 4:]
                truncated = f"{head}\n\n... [{len(content) - len(head) - len(tail)} chars truncated] ...\n\n{tail}"
                result.append({**msg, "content": truncated})
                continue

            # Remove empty assistant messages
            if msg.get("role") == "assistant" and not content and not msg.get("tool_calls"):
                continue

            # Remove old noise
            if isinstance(content, str) and content.strip() in ("", "OK", "ok", "Done", "done"):
                if i < len(messages) - self.preserve_last_n:
                    continue

            result.append(msg)

        return result


def _msg_hash(msg: dict) -> str:
    """Hash a message for dedup.

    Includes tool_call_id and tool_calls to avoid collapsing
    distinct tool results that happen to share the same content.
    """
    parts = [
        str(msg.get("role", "")),
        str(msg.get("content", "")),
        str(msg.get("tool_call_id", "")),
    ]
    # Include tool_call IDs so assistant messages with different calls aren't deduped
    for tc in msg.get("tool_calls") or []:
        if isinstance(tc, dict):
            parts.append(tc.get("id", ""))
        else:
            parts.append(getattr(tc, "id", "") or "")
    return hashlib.md5("|".join(parts).encode()).hexdigest()[:12]


def _normalize_whitespace(text: str) -> str:
    """Collapse multiple blank lines, strip trailing spaces."""
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +\n', '\n', text)
    return text.strip()
