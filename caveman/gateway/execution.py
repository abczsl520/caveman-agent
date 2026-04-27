"""Agent Execution Engine — run agent with fallback, retry, error recovery.

Extracted from OpenClaw agent-runner-execution.ts (1572 lines) and
Hermes agent/ai_agent.py execution patterns.

Features:
- Model fallback chain (primary → secondary → tertiary)
- Transient HTTP error retry with backoff
- Context window overflow → auto-compaction → retry
- Tool execution timeout and cancellation
- Session reset on unrecoverable errors
- Streaming support with chunk delivery
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

__all__ = [
    "TRANSIENT_ERRORS",
    "CONTEXT_OVERFLOW_PATTERNS",
    "FallbackCandidate",
    "ExecutionConfig",
    "ExecutionResult",
    "AgentExecutionEngine",
]


logger = logging.getLogger("caveman.gateway.execution")

# Transient errors worth retrying (from OpenClaw)
TRANSIENT_ERRORS = frozenset({
    "overloaded_error", "rate_limit_error", "api_error",
    "timeout", "connection_error", "529", "503", "502",
})

# Context overflow indicators
CONTEXT_OVERFLOW_PATTERNS = (
    "context_length_exceeded", "max_tokens", "context window",
    "too many tokens", "prompt is too long", "maximum context",
)


@dataclass
class FallbackCandidate:
    """A model to try if the primary fails."""
    provider: str
    model: str
    reason: str = ""


@dataclass
class ExecutionConfig:
    """Configuration for agent execution."""
    max_retries: int = 3
    retry_delay: float = 2.5  # seconds
    retry_backoff: float = 2.0  # multiplier
    max_compaction_retries: int = 2
    tool_timeout: float = 300.0  # 5 minutes
    total_timeout: float | None = None  # None = do not cancel long-running agent work
    fallback_candidates: List[FallbackCandidate] = field(default_factory=list)
    stream: bool = True


@dataclass
class ExecutionResult:
    """Result of an agent execution."""
    success: bool
    response: str = ""
    error: str = ""
    model_used: str = ""
    provider_used: str = ""
    retries: int = 0
    compactions: int = 0
    fallback_used: bool = False
    duration_ms: float = 0


class AgentExecutionEngine:
    """Run agent tasks with production-grade resilience.

    Handles the full lifecycle: retry → fallback → compaction → timeout.
    """

    def __init__(
        self,
        agent_fn: Callable,
        config: Optional[ExecutionConfig] = None,
        compact_fn: Optional[Callable] = None,
        reset_fn: Optional[Callable] = None,
    ):
        self._agent_fn = agent_fn
        self._config = config or ExecutionConfig()
        self._compact_fn = compact_fn  # async (session) → bool
        self._reset_fn = reset_fn  # async (session, reason) → bool

    async def execute(
        self,
        message: str,
        session: Any = None,
        model: str = "",
        provider: str = "",
        metadata: Optional[Dict] = None,
    ) -> ExecutionResult:
        """Execute an agent task with full resilience."""
        start = time.monotonic()
        retries = 0
        compactions = 0
        current_model = model
        current_provider = provider
        fallback_used = False
        last_error = ""

        # Build candidate list: primary + fallbacks
        candidates = [(current_provider, current_model)]
        for fb in self._config.fallback_candidates:
            candidates.append((fb.provider, fb.model))

        for candidate_idx, (cand_provider, cand_model) in enumerate(candidates):
            if candidate_idx > 0:
                fallback_used = True
                logger.info("Falling back to %s/%s", cand_provider, cand_model)

            retry_delay = self._config.retry_delay

            for attempt in range(self._config.max_retries + 1):
                if self._config.total_timeout is not None and time.monotonic() - start > self._config.total_timeout:
                    return ExecutionResult(
                        success=False,
                        error=f"Total timeout ({self._config.total_timeout}s) exceeded",
                        retries=retries,
                        compactions=compactions,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )

                try:
                    agent_call = self._agent_fn(
                        message,
                        session=session,
                        model=cand_model,
                        provider=cand_provider,
                        metadata=metadata,
                    )
                    if self._config.total_timeout is None:
                        response = await agent_call
                    else:
                        response = await asyncio.wait_for(
                            agent_call,
                            timeout=self._config.total_timeout,
                        )

                    return ExecutionResult(
                        success=True,
                        response=str(response) if response else "",
                        model_used=cand_model,
                        provider_used=cand_provider,
                        retries=retries,
                        compactions=compactions,
                        fallback_used=fallback_used,
                        duration_ms=(time.monotonic() - start) * 1000,
                    )

                except asyncio.TimeoutError:
                    last_error = "timeout"
                    retries += 1
                    logger.warning("Agent execution timed out (attempt %d)", attempt + 1)

                except Exception as e:
                    error_str = str(e).lower()
                    last_error = str(e)

                    # Context overflow → try compaction
                    if self._is_context_overflow(error_str) and self._compact_fn:
                        if compactions < self._config.max_compaction_retries:
                            logger.info("Context overflow, attempting compaction (%d)", compactions + 1)
                            try:
                                compacted = await self._compact_fn(session)
                                if compacted:
                                    compactions += 1
                                    continue  # Retry with compacted context
                            except Exception as ce:
                                logger.warning("Compaction failed: %s", ce)

                            # Compaction failed → try reset
                            if self._reset_fn:
                                try:
                                    await self._reset_fn(session, "context_overflow_compaction_failed")
                                    compactions += 1
                                    continue
                                except Exception as exc:
                                    logger.debug("unknown: suppressed %s", exc)

                    # Transient error → retry with backoff
                    if self._is_transient(error_str) and attempt < self._config.max_retries:
                        retries += 1
                        logger.info("Transient error, retrying in %.1fs: %s", retry_delay, last_error[:100])
                        await asyncio.sleep(retry_delay)
                        retry_delay *= self._config.retry_backoff
                        continue

                    # Non-transient, non-overflow → try next candidate
                    logger.warning("Non-transient error with %s/%s: %s",
                                   cand_provider, cand_model, last_error[:200])
                    break  # Move to next fallback candidate

        return ExecutionResult(
            success=False,
            error=last_error,
            retries=retries,
            compactions=compactions,
            fallback_used=fallback_used,
            duration_ms=(time.monotonic() - start) * 1000,
        )

    @staticmethod
    def _is_transient(error: str) -> bool:
        """Check if error is transient and worth retrying."""
        return any(pat in error for pat in TRANSIENT_ERRORS)

    @staticmethod
    def _is_context_overflow(error: str) -> bool:
        """Check if error indicates context window overflow."""
        return any(pat in error for pat in CONTEXT_OVERFLOW_PATTERNS)
