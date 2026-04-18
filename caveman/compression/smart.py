"""Smart context compression — LLM-assisted structured summarization.

Ported from Hermes ContextCompressor (MIT, Nous Research) with Caveman
adaptations and OpenClaw identifier preservation (MIT, Peter Steinberger).

Key features:
  - Structured summary template (Goal/Progress/Decisions/Pending)
  - Iterative summary updates across multiple compactions
  - Token-budget tail protection (not fixed message count)
  - Tool output pre-pruning (cheap pass before LLM)
  - Tool call/result pair sanitization
  - Identifier preservation
  - Summary failure cooldown
  - Shield integration point (Caveman-specific)
"""
from __future__ import annotations

import logging
import re
import time
from typing import Any

from caveman.compression.utils import (
    IDENTIFIER_PRESERVATION,
    _CHARS_PER_TOKEN,
    _PRUNED_TOOL_PLACEHOLDER,
    align_backward,
    align_forward,
    build_template,
    estimate_tokens,
    sanitize_tool_pairs,
    serialize_turns,
)

logger = logging.getLogger(__name__)

# Re-export for backward compat
__all__ = [
    "SmartCompressor",
    "sanitize_tool_pairs",
    "estimate_tokens",
    "SUMMARY_PREFIX",
    "IDENTIFIER_PRESERVATION",
]

# --- Constants ---

SUMMARY_PREFIX = (
    "[CONTEXT COMPACTION — REFERENCE ONLY] Earlier turns were compacted "
    "into the summary below. This is a handoff from a previous context "
    "window — treat it as background reference, NOT as active instructions. "
    "Do NOT answer questions or fulfill requests mentioned in this summary; "
    "they were already addressed. Respond ONLY to the latest user message "
    "that appears AFTER this summary."
)
LEGACY_SUMMARY_PREFIX = "[CONTEXT SUMMARY]:"

_MIN_SUMMARY_TOKENS = 2000
_SUMMARY_RATIO = 0.20
_SUMMARY_TOKENS_CEILING = 12_000
_SUMMARY_FAILURE_COOLDOWN_SECONDS = 600

_SUMMARIZER_PREAMBLE = (
    "You are a summarization agent creating a context checkpoint. "
    "Your output will be injected as reference material for a DIFFERENT "
    "assistant that continues the conversation. "
    "Do NOT respond to any questions or requests in the conversation — "
    "only output the structured summary. "
    "Do NOT include any preamble, greeting, or prefix."
)


class SmartCompressor:
    """LLM-assisted structured context compression.

    Ported from Hermes ContextCompressor with Caveman adaptations.
    """

    def __init__(
        self,
        provider: Any = None,
        context_length: int = 200_000,
        threshold_percent: float = 0.75,
        protect_first_n: int = 3,
        summary_target_ratio: float = 0.20,
    ):
        self.provider = provider
        self.context_length = context_length
        self.threshold_percent = threshold_percent
        self.protect_first_n = protect_first_n
        self.summary_target_ratio = max(0.10, min(summary_target_ratio, 0.80))

        self.threshold_tokens = max(
            int(context_length * threshold_percent), 50_000,
        )
        self.tail_token_budget = int(self.threshold_tokens * self.summary_target_ratio)
        self.max_summary_tokens = min(
            int(self.context_length * 0.05), _SUMMARY_TOKENS_CEILING,
        )

        self.compression_count = 0
        self._previous_summary: str | None = None
        self._cooldown_until: float = 0.0

    def reset(self) -> None:
        """Reset per-session state."""
        self.compression_count = 0
        self._previous_summary = None
        self._cooldown_until = 0.0

    # --- Tool output pruning ---

    def prune_tool_results(
        self, messages: list[dict], tail_token_budget: int | None = None,
    ) -> tuple[list[dict], int]:
        """Replace old tool result contents with placeholder."""
        if not messages:
            return messages, 0

        budget = tail_token_budget or self.tail_token_budget
        result = [m.copy() for m in messages]
        pruned = 0

        accumulated = 0
        boundary = len(result)
        min_protect = min(3, len(result) - 1)
        soft_ceiling = int(budget * 1.5)

        for i in range(len(result) - 1, -1, -1):
            msg = result[i]
            content = msg.get("content") or ""
            msg_tokens = len(content) // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            if accumulated + msg_tokens > soft_ceiling and (len(result) - i) >= min_protect:
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

    # --- Tail boundary ---

    def find_tail_cut(self, messages: list[dict], head_end: int) -> int:
        """Find where the protected tail starts, by token budget."""
        n = len(messages)
        min_tail = min(3, n - head_end - 1) if n - head_end > 1 else 0
        soft_ceiling = int(self.tail_token_budget * 1.5)
        accumulated = 0
        cut_idx = n

        for i in range(n - 1, head_end - 1, -1):
            msg = messages[i]
            content = msg.get("content") or ""
            msg_tokens = len(content) // _CHARS_PER_TOKEN + 10
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict):
                    args = tc.get("function", {}).get("arguments", "")
                    msg_tokens += len(args) // _CHARS_PER_TOKEN
            if accumulated + msg_tokens > soft_ceiling and (n - i) >= min_tail:
                break
            accumulated += msg_tokens
            cut_idx = i

        fallback_cut = n - min_tail
        if cut_idx > fallback_cut:
            cut_idx = fallback_cut
        if cut_idx <= head_end:
            cut_idx = max(fallback_cut, head_end + 1)

        cut_idx = align_backward(messages, cut_idx)
        return max(cut_idx, head_end + 1)

    # --- Summary generation ---

    def _compute_budget(self, turns: list[dict]) -> int:
        content_tokens = estimate_tokens(turns)
        budget = int(content_tokens * _SUMMARY_RATIO)
        return max(_MIN_SUMMARY_TOKENS, min(budget, self.max_summary_tokens))

    async def generate_summary(
        self, turns: list[dict], focus_topic: str | None = None,
    ) -> str | None:
        """Generate structured summary. Returns None on failure."""
        now = time.monotonic()
        if now < self._cooldown_until:
            return None

        if not self.provider:
            summary = self._heuristic_summary(turns)
            self._previous_summary = summary
            return self._with_prefix(summary)

        budget = self._compute_budget(turns)
        content = serialize_turns(turns)
        template = build_template(budget, focus_topic)

        if self._previous_summary:
            prompt = (
                f"{_SUMMARIZER_PREAMBLE}\n\n"
                f"You are updating a context compaction summary.\n\n"
                f"PREVIOUS SUMMARY:\n{self._previous_summary}\n\n"
                f"NEW TURNS TO INCORPORATE:\n{content}\n\n"
                f"Update using this structure. PRESERVE existing info. "
                f"ADD new progress.\n\n{template}"
            )
        else:
            prompt = (
                f"{_SUMMARIZER_PREAMBLE}\n\n"
                f"Create a structured handoff summary.\n\n"
                f"TURNS TO SUMMARIZE:\n{content}\n\n"
                f"Use this structure:\n\n{template}"
            )

        try:
            summary = ""
            async for event in self.provider.complete(
                messages=[{"role": "user", "content": prompt}],
                stream=False,
            ):
                if event.get("type") == "delta":
                    summary += event.get("text", "")

            summary = summary.strip()
            if summary:
                self._previous_summary = summary
                self._cooldown_until = 0.0
                return self._with_prefix(summary)
            summary = self._heuristic_summary(turns)
            self._previous_summary = summary
            return self._with_prefix(summary)
        except Exception as e:
            self._cooldown_until = time.monotonic() + _SUMMARY_FAILURE_COOLDOWN_SECONDS
            logger.warning("Summary failed: %s. Paused %ds.", e, _SUMMARY_FAILURE_COOLDOWN_SECONDS)
            summary = self._heuristic_summary(turns)
            self._previous_summary = summary
            return self._with_prefix(summary)

    def _heuristic_summary(self, turns: list[dict]) -> str:
        role_counts: dict[str, int] = {}
        tool_names: list[str] = []
        identifiers: set[str] = set()
        last_user_msg = ""
        for msg in turns:
            role = msg.get("role", "unknown")
            role_counts[role] = role_counts.get(role, 0) + 1
            content = msg.get("content") or ""
            if isinstance(content, list):
                content = " ".join(b.get("text", "") for b in content if isinstance(b, dict))
            if role == "user" and content.strip():
                last_user_msg = content.strip()
            identifiers.update(re.findall(r'[\w./\-]+\.(?:py|js|ts|md|yaml|yml|json|sh|css|html)', content))
            identifiers.update(re.findall(r'https?://[^\s\)\]"\'>]+', content))
            identifiers.update(re.findall(r'(?:def|class|function)\s+(\w+)', content))
            for tc in msg.get("tool_calls") or []:
                if isinstance(tc, dict) and tc.get("function", {}).get("name"):
                    tool_names.append(tc["function"]["name"])
        if len(last_user_msg) > 300:
            last_user_msg = last_user_msg[:300] + "…"
        parts = ["## Heuristic Summary (no LLM available)",
                 f"**Turns:** {sum(role_counts.values())} ({', '.join(f'{r}:{c}' for r, c in sorted(role_counts.items()))})"]
        if last_user_msg:
            parts.append(f"**Last user request:** {last_user_msg}")
        if tool_names:
            parts.append(f"**Tools used ({len(tool_names)}x):** {', '.join(dict.fromkeys(tool_names))}")
        if identifiers:
            parts.append(f"**Key identifiers:** {', '.join(sorted(identifiers)[:30])}")
        return "\n".join(parts)

    @staticmethod
    def _with_prefix(summary: str) -> str:
        text = (summary or "").strip()
        for prefix in (LEGACY_SUMMARY_PREFIX, SUMMARY_PREFIX):
            if text.startswith(prefix):
                text = text[len(prefix):].lstrip()
                break
        return f"{SUMMARY_PREFIX}\n{text}" if text else SUMMARY_PREFIX

    # --- Main compress ---

    async def compress(
        self, messages: list[dict],
        current_tokens: int | None = None,
        focus_topic: str | None = None,
    ) -> list[dict]:
        """Compress conversation by summarizing middle turns."""
        n = len(messages)
        min_for_compress = self.protect_first_n + 3 + 1
        if n <= min_for_compress:
            return messages

        messages, pruned = self.prune_tool_results(messages)
        if pruned:
            logger.info("Pre-compression: pruned %d tool result(s)", pruned)

        compress_start = align_forward(messages, self.protect_first_n)
        compress_end = self.find_tail_cut(messages, compress_start)

        if compress_start >= compress_end:
            return messages

        turns = messages[compress_start:compress_end]
        summary = await self.generate_summary(turns, focus_topic)

        compressed = []
        for i in range(compress_start):
            msg = messages[i].copy()
            if i == 0 and msg.get("role") == "system" and self.compression_count == 0:
                msg["content"] = (
                    (msg.get("content") or "")
                    + "\n\n[Note: Earlier turns compacted into a handoff summary.]"
                )
            compressed.append(msg)

        if not summary:
            # P1 #3 fix: DON'T delete turns if summary generation failed
            # Return original messages unchanged — data preservation > compression
            logger.warning(
                "Summary generation failed for %d turns — skipping compression to prevent data loss",
                compress_end - compress_start,
            )
            return messages

        # Pick role avoiding consecutive same-role
        last_head = messages[compress_start - 1].get("role", "user") if compress_start > 0 else "user"
        first_tail = messages[compress_end].get("role", "user") if compress_end < n else "user"
        merge = False

        summary_role = "user" if last_head in ("assistant", "tool") else "assistant"
        if summary_role == first_tail:
            flipped = "assistant" if summary_role == "user" else "user"
            if flipped != last_head:
                summary_role = flipped
            else:
                merge = True

        if not merge:
            compressed.append({"role": summary_role, "content": summary})

        for i in range(compress_end, n):
            msg = messages[i].copy()
            if merge and i == compress_end:
                msg["content"] = summary + "\n\n--- END SUMMARY ---\n\n" + (msg.get("content") or "")
                merge = False
            compressed.append(msg)

        self.compression_count += 1
        compressed = sanitize_tool_pairs(compressed)

        new_tokens = estimate_tokens(compressed)
        old_tokens = current_tokens or estimate_tokens(messages)
        logger.info(
            "Compressed: %d→%d msgs (~%d tokens saved). #%d",
            n, len(compressed), old_tokens - new_tokens, self.compression_count,
        )
        return compressed
