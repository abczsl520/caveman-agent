"""Agent Runner Memory — compaction, memory flush, token tracking.

Extracted from OpenClaw agent-runner-memory.ts (848 lines).
Manages context window budget, triggers compaction, and flushes memories.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Optional

logger = logging.getLogger("caveman.gateway.agent_memory")


@dataclass
class MemoryFlushPlan:
    """Configuration for when to flush/compact memory."""
    soft_threshold_tokens: int = 4000
    reserve_tokens_floor: int = 20000
    compact_trigger_ratio: float = 0.8  # Compact when usage > 80% of context
    flush_on_idle_seconds: float = 300  # Flush after 5 min idle
    max_session_tokens: int = 0  # 0 = use model default


@dataclass
class TokenBudget:
    """Token budget tracking for a session."""
    context_window: int = 200000
    reserve_floor: int = 20000
    soft_threshold: int = 4000
    current_prompt_tokens: int = 0
    current_completion_tokens: int = 0
    total_tokens: int = 0
    compaction_count: int = 0
    last_compaction_at: float = 0

    @property
    def available(self) -> int:
        return max(0, self.context_window - self.reserve_floor - self.total_tokens)

    @property
    def usage_ratio(self) -> float:
        if self.context_window <= 0:
            return 0
        return self.total_tokens / self.context_window

    @property
    def should_compact(self) -> bool:
        threshold = self.context_window - self.reserve_floor - self.soft_threshold
        return self.total_tokens > threshold > 0

    def update(self, prompt_tokens: int = 0, completion_tokens: int = 0) -> None:
        self.current_prompt_tokens = prompt_tokens
        self.current_completion_tokens = completion_tokens
        self.total_tokens = prompt_tokens + completion_tokens

    def record_compaction(self, tokens_after: int = 0) -> None:
        self.compaction_count += 1
        self.last_compaction_at = time.monotonic()
        if tokens_after > 0:
            self.total_tokens = tokens_after


def resolve_context_window(model: str, config: Optional[Dict] = None) -> int:
    """Resolve context window size for a model."""
    # Config override
    if config:
        override = config.get("context_tokens")
        if isinstance(override, int) and override > 0:
            return override

    # Delegate to depth module for model lookup; use 200k default for unknown models
    from caveman.gateway.agent_memory_depth import MODEL_CONTEXT_WINDOWS
    for prefix in MODEL_CONTEXT_WINDOWS:
        if model.startswith(prefix):
            return get_context_window(model)

    return 200000  # Safe default for unknown models


class AgentMemoryManager:
    """Manages agent memory lifecycle: compaction, flush, token tracking."""

    def __init__(
        self,
        compact_fn: Optional[Callable] = None,
        flush_fn: Optional[Callable] = None,
        config: Optional[Dict] = None,
    ):
        self._compact_fn = compact_fn
        self._flush_fn = flush_fn
        self._config = config or {}
        self._budgets: Dict[str, TokenBudget] = {}
        self._plan = MemoryFlushPlan(
            **{k: v for k, v in self._config.get("memory_flush", {}).items()
               if k in MemoryFlushPlan.__dataclass_fields__}
        )

    def get_budget(self, session_key: str, model: str = "") -> TokenBudget:
        """Get or create token budget for a session."""
        if session_key not in self._budgets:
            window = resolve_context_window(model, self._config)
            self._budgets[session_key] = TokenBudget(
                context_window=window,
                reserve_floor=self._plan.reserve_tokens_floor,
                soft_threshold=self._plan.soft_threshold_tokens,
            )
        return self._budgets[session_key]

    def update_usage(self, session_key: str, prompt_tokens: int, completion_tokens: int) -> None:
        """Update token usage after an API call."""
        budget = self._budgets.get(session_key)
        if budget:
            budget.update(prompt_tokens, completion_tokens)

    async def check_compaction(self, session_key: str, model: str = "") -> bool:
        """Check if compaction is needed and run it if so."""
        budget = self.get_budget(session_key, model)
        if not budget.should_compact:
            return False

        if not self._compact_fn:
            logger.warning("Compaction needed for %s but no compact_fn", session_key)
            return False

        logger.info(
            "Triggering compaction for %s (tokens=%d, window=%d, ratio=%.2f)",
            session_key, budget.total_tokens, budget.context_window, budget.usage_ratio,
        )

        try:
            result = self._compact_fn(session_key)
            if hasattr(result, "__await__"):
                result = await result

            tokens_after = result.get("tokens_after", 0) if isinstance(result, dict) else 0
            budget.record_compaction(tokens_after)
            return True

        except Exception as e:
            logger.error("Compaction failed for %s: %s", session_key, e)
            return False

    async def flush_if_needed(self, session_key: str) -> bool:
        """Flush memory if idle timeout reached."""
        if not self._flush_fn:
            return False

        budget = self._budgets.get(session_key)
        if not budget:
            return False

        # Only flush if we have meaningful content
        if budget.total_tokens < 1000:
            return False

        try:
            result = self._flush_fn(session_key)
            if hasattr(result, "__await__"):
                result = await result
            return bool(result)
        except Exception as e:
            logger.error("Memory flush failed for %s: %s", session_key, e)
            return False

    def remove_session(self, session_key: str) -> None:
        self._budgets.pop(session_key, None)

    def get_stats(self) -> Dict[str, Any]:
        return {
            "sessions": len(self._budgets),
            "total_tokens": sum(b.total_tokens for b in self._budgets.values()),
            "compactions": sum(b.compaction_count for b in self._budgets.values()),
        }
from caveman.gateway.agent_memory_depth import (  # noqa: F401  # depth wiring
    estimate_tokens_for_model,
    estimate_transcript_tokens,
    MODEL_CONTEXT_WINDOWS,
    get_context_window,
    should_compact,
    MemoryFlushConfig,
    flush_transcript,
    load_transcript,
    CompactionResult,
    prepare_compaction,
)

__all__ = [
    "MemoryFlushPlan",
    "TokenBudget",
    "resolve_context_window",
    "AgentMemoryManager",
    "estimate_tokens_for_model",
    "estimate_transcript_tokens",
    "MODEL_CONTEXT_WINDOWS",
    "get_context_window",
    "should_compact",
    "MemoryFlushConfig",
    "flush_transcript",
    "load_transcript",
    "CompactionResult",
    "prepare_compaction",
]

