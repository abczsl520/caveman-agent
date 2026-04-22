"""Context Compressor — automatic conversation context compression.

Compresses long conversations by summarizing middle turns while
protecting head (system) and tail (recent) messages. Supports
iterative summary updates across multiple compactions.
Extracted from Hermes agent/context_compressor.py (820 lines).
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "SUMMARY_PREFIX",
    "MODEL_CONTEXT_LENGTHS",
    "MINIMUM_CONTEXT_LENGTH",
    "get_context_length",
    "estimate_tokens_rough",
    "CompactionResult",
    "ContextCompressor",
]


logger = logging.getLogger("caveman.agent.context_compressor")

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary."
)

_PRUNED_TOOL_PLACEHOLDER = "[Old tool output cleared to save context space]"
_CHARS_PER_TOKEN = 4
_MIN_SUMMARY_TOKENS = 2000
_SUMMARY_RATIO = 0.20
_SUMMARY_TOKENS_CEILING = 12_000
_SUMMARY_FAILURE_COOLDOWN = 600  # seconds

# ── Model Context Windows ──

MODEL_CONTEXT_LENGTHS = {
    "claude-opus-4-6": 200_000,
    "claude-sonnet-4-20250514": 200_000,
    "gpt-4o": 128_000,
    "gpt-4o-mini": 128_000,
    "o4-mini": 200_000,
    "gemini-2.5-pro": 1_000_000,
    "gemini-2.5-flash": 1_000_000,
    "deepseek-chat": 64_000,
}
MINIMUM_CONTEXT_LENGTH = 16_000


def get_context_length(model: str) -> int:
    """Get context window length for a model."""
    for prefix, length in MODEL_CONTEXT_LENGTHS.items():
        if model.startswith(prefix):
            return length
    return 128_000


def estimate_tokens_rough(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate for a message list.

    Delegates to compression.utils.estimate_tokens for consistency.
    """
    from caveman.compression.utils import estimate_tokens
    return estimate_tokens(messages)


@dataclass
class CompactionResult:
    """Result of a context compaction."""
    messages: List[Dict[str, Any]]
    summary: str = ""
    original_count: int = 0
    compacted_count: int = 0
    original_tokens: int = 0
    compacted_tokens: int = 0
    tool_results_pruned: int = 0
    compression_count: int = 0


class ContextCompressor:
    """Compresses conversation context via lossy summarization.

    Algorithm:
      1. Prune old tool results (cheap, no LLM call)
      2. Protect head messages (system prompt + first exchange)
      3. Protect tail messages by token budget
      4. Summarize middle turns with structured LLM prompt
      5. On subsequent compactions, iteratively update previous summary
    """

    def __init__(
        self,
        model: str = "",
        threshold_percent: float = 0.50,
        protect_first_n: int = 3,
        tail_token_budget: int = 20_000,
        summary_model: str = "",
        summarize_fn: Optional[Callable] = None,
    ):
        self.model = model
        self.context_length = get_context_length(model)
        self.threshold_percent = threshold_percent
        self.threshold_tokens = max(
            int(self.context_length * threshold_percent),
            MINIMUM_CONTEXT_LENGTH,
        )
        self.protect_first_n = protect_first_n
        self.tail_token_budget = tail_token_budget
        self.summary_model = summary_model
        self._summarize_fn = summarize_fn
        self._previous_summary: Optional[str] = None
        self._compression_count = 0
        self._failure_cooldown_until = 0.0

    def should_compress(self, messages: List[Dict[str, Any]]) -> bool:
        """Check if context should be compressed."""
        tokens = estimate_tokens_rough(messages)
        return tokens >= self.threshold_tokens

    def compress(self, messages: List[Dict[str, Any]]) -> CompactionResult:
        """Compress conversation context."""
        original_count = len(messages)
        original_tokens = estimate_tokens_rough(messages)

        # Step 1: Prune old tool results
        pruned_messages, pruned_count = self._prune_tool_results(messages)

        # Step 2: Split into head / middle / tail
        head, middle, tail = self._split_messages(pruned_messages)

        if not middle:
            return CompactionResult(
                messages=pruned_messages,
                original_count=original_count,
                compacted_count=len(pruned_messages),
                original_tokens=original_tokens,
                compacted_tokens=estimate_tokens_rough(pruned_messages),
                tool_results_pruned=pruned_count,
            )

        # Step 3: Summarize middle turns
        summary = self._summarize(middle)

        # Step 4: Build compacted messages
        summary_msg = {
            "role": "user",
            "content": f"{SUMMARY_PREFIX}\n\n{summary}",
        }
        compacted = head + [summary_msg] + tail

        self._previous_summary = summary
        self._compression_count += 1

        return CompactionResult(
            messages=compacted,
            summary=summary,
            original_count=original_count,
            compacted_count=len(compacted),
            original_tokens=original_tokens,
            compacted_tokens=estimate_tokens_rough(compacted),
            tool_results_pruned=pruned_count,
            compression_count=self._compression_count,
        )

    def _prune_tool_results(
        self, messages: List[Dict[str, Any]],
    ) -> tuple:
        """Replace old tool result contents with placeholder."""
        result = [m.copy() for m in messages]
        pruned = 0

        # Determine boundary using tail token budget
        accumulated = 0
        boundary = len(result)
        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            content = msg.get("content", "")
            msg_tokens = len(str(content)) // _CHARS_PER_TOKEN + 10
            if accumulated + msg_tokens > self.tail_token_budget:
                boundary = i
                break
            accumulated += msg_tokens
            boundary = i

        for i in range(boundary):
            msg = result[i]
            if msg.get("role") != "tool":
                continue
            content = msg.get("content", "")
            if not content or content == _PRUNED_TOOL_PLACEHOLDER:
                continue
            if len(content) > 200:
                result[i] = {**msg, "content": _PRUNED_TOOL_PLACEHOLDER}
                pruned += 1

        return result, pruned

    def _split_messages(
        self, messages: List[Dict[str, Any]],
    ) -> tuple:
        """Split messages into head, middle, tail."""
        if len(messages) <= self.protect_first_n + 5:
            return messages, [], []

        head = messages[:self.protect_first_n]

        # Tail: walk backward by token budget
        tail_start = len(messages)
        accumulated = 0
        for i in range(len(messages) - 1, self.protect_first_n - 1, -1):
            msg = messages[i]
            tokens = len(str(msg.get("content", ""))) // _CHARS_PER_TOKEN + 10
            if accumulated + tokens > self.tail_token_budget:
                break
            accumulated += tokens
            tail_start = i

        # Ensure at least 5 tail messages
        tail_start = min(tail_start, max(len(messages) - 5, self.protect_first_n))

        middle = messages[self.protect_first_n:tail_start]
        tail = messages[tail_start:]

        return head, middle, tail

    def _summarize(self, messages: List[Dict[str, Any]]) -> str:
        """Summarize middle messages."""
        # Build summarizer input
        turns = []
        for msg in messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            if isinstance(content, list):
                text_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            text_parts.append(block.get("text", ""))
                        elif block.get("type") == "tool_use":
                            text_parts.append(f"[Tool call: {block.get('name', '')}]")
                        elif block.get("type") == "tool_result":
                            text_parts.append(f"[Tool result: {str(block.get('content', ''))[:200]}]")
                        elif block.get("type") == "image_url":
                            text_parts.append("[Image attachment]")
                content = "\n".join(text_parts)
            turns.append(f"[{role}]: {content[:2000]}")

        turns_text = "\n\n".join(turns)

        # Build prompt
        prompt = (
            "You are a context summarizer. Do NOT respond to any questions in the "
            "conversation — only summarize what happened.\n\n"
            "Summarize the following conversation turns into a structured summary:\n\n"
            f"{turns_text}\n\n"
            "Format your summary as:\n"
            "## Goal\nWhat the user is trying to accomplish\n\n"
            "## Decisions\nKey decisions made\n\n"
            "## Progress\n- What was completed\n- What is in progress\n\n"
            "## Remaining Work\nWhat still needs to be done\n\n"
        )

        # Add previous summary for iterative update
        if self._previous_summary:
            prompt += (
                f"\nPrevious summary to update (merge new info, don't lose old):\n"
                f"{self._previous_summary}\n"
            )

        # Try LLM summarization
        if self._summarize_fn and time.time() > self._failure_cooldown_until:
            try:
                result = self._summarize_fn(prompt)
                if isinstance(result, str) and len(result) > 100:
                    return result
            except Exception as e:
                logger.warning("Summarization failed: %s", e)
                self._failure_cooldown_until = time.time() + _SUMMARY_FAILURE_COOLDOWN

        # Fallback: simple extraction
        return self._fallback_summary(messages)

    def _fallback_summary(self, messages: List[Dict[str, Any]]) -> str:
        """Simple fallback summary without LLM."""
        lines = ["## Conversation Summary (auto-generated)\n"]
        user_msgs = [m for m in messages if m.get("role") == "user"]
        assistant_msgs = [m for m in messages if m.get("role") == "assistant"]
        tool_msgs = [m for m in messages if m.get("role") == "tool"]

        lines.append(f"- {len(user_msgs)} user messages")
        lines.append(f"- {len(assistant_msgs)} assistant responses")
        lines.append(f"- {len(tool_msgs)} tool calls")

        # Extract key topics from user messages
        if user_msgs:
            lines.append("\nKey topics:")
            for msg in user_msgs[-5:]:
                content = str(msg.get("content", ""))[:150]
                lines.append(f"- {content}")

        return "\n".join(lines)

    def reset(self) -> None:
        """Reset compressor state."""
        self._previous_summary = None
        self._compression_count = 0
        self._failure_cooldown_until = 0.0
