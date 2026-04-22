"""Agent Runner — core agent execution orchestration.

Extracted from OpenClaw agent-runner.ts (874 lines).
Orchestrates: session lookup, memory preflight, agent dispatch, reply routing.
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

logger = logging.getLogger("caveman.gateway.agent_runner")


@dataclass
class RunContext:
    """Context for an agent run."""
    session_key: str = ""
    session_id: str = ""
    command_body: str = ""
    model: str = ""
    provider: str = ""
    is_new_session: bool = False
    is_heartbeat: bool = False
    is_streaming: bool = False
    verbose_level: str = "off"
    reasoning_mode: str = "off"
    reset_triggered: bool = False
    typing_mode: str = "auto"  # auto | always | never
    block_streaming: bool = False


@dataclass
class RunResult:
    """Result of an agent run."""
    ok: bool = True
    text: str = ""
    tool_calls: int = 0
    tokens_prompt: int = 0
    tokens_completion: int = 0
    duration_ms: float = 0
    compacted: bool = False
    error: str = ""
    phase: str = "completed"  # completed | error | timeout | cancelled


class AgentRunner:
    """Orchestrates agent execution with full lifecycle management."""

    def __init__(
        self,
        agent_fn: Optional[Callable] = None,
        memory_manager: Optional[Any] = None,
        session_manager: Optional[Any] = None,
        typing_fn: Optional[Callable] = None,
    ):
        self._agent_fn = agent_fn
        self._memory = memory_manager
        self._sessions = session_manager
        self._typing_fn = typing_fn
        self._active_runs: Dict[str, asyncio.Task] = {}
        self._cancel_events: Dict[str, asyncio.Event] = {}

    # ── Run ──

    async def run(self, ctx: RunContext) -> RunResult:
        """Execute a full agent run."""
        start = time.monotonic()
        result = RunResult()

        # Cancel any existing run for this session
        await self._cancel_existing(ctx.session_key)

        # Create cancel event
        cancel = asyncio.Event()
        self._cancel_events[ctx.session_key] = cancel

        try:
            # 1. Reset if triggered
            if ctx.reset_triggered and self._sessions:
                try:
                    reset_fn = getattr(self._sessions, "reset", None)
                    if reset_fn:
                        r = reset_fn(ctx.session_key)
                        if hasattr(r, "__await__"):
                            await r
                    ctx.is_new_session = True
                except Exception as e:
                    logger.warning("Reset failed: %s", e)

            # 2. Memory preflight (compaction check)
            if self._memory and not ctx.is_heartbeat:
                try:
                    compacted = await self._memory.check_compaction(ctx.session_key, ctx.model)
                    result.compacted = compacted
                except Exception as e:
                    logger.warning("Memory preflight failed: %s", e)

            # 3. Start typing
            if self._typing_fn and ctx.typing_mode != "never":
                try:
                    t = self._typing_fn(ctx.session_key)
                    if hasattr(t, "__await__"):
                        await t
                except Exception as exc:
                    logger.debug("run: suppressed %s", exc)

            # 4. Agent dispatch
            if not self._agent_fn:
                result.ok = False
                result.error = "No agent function configured"
                result.phase = "error"
                return result

            if cancel.is_set():
                result.phase = "cancelled"
                return result

            try:
                agent_result = self._agent_fn(ctx)
                if hasattr(agent_result, "__await__"):
                    agent_result = await agent_result

                if isinstance(agent_result, dict):
                    result.text = agent_result.get("text", "")
                    result.tool_calls = agent_result.get("tool_calls", 0)
                    result.tokens_prompt = agent_result.get("prompt_tokens", 0)
                    result.tokens_completion = agent_result.get("completion_tokens", 0)
                elif isinstance(agent_result, str):
                    result.text = agent_result
                else:
                    result.text = str(agent_result) if agent_result else ""

            except asyncio.CancelledError:
                result.phase = "cancelled"
            except Exception as e:
                result.ok = False
                result.error = str(e)
                result.phase = "error"
                logger.error("Agent run failed: %s", e)

            # 5. Update memory usage
            if self._memory and result.tokens_prompt > 0:
                self._memory.update_usage(
                    ctx.session_key, result.tokens_prompt, result.tokens_completion,
                )

        finally:
            self._cancel_events.pop(ctx.session_key, None)
            self._active_runs.pop(ctx.session_key, None)
            result.duration_ms = (time.monotonic() - start) * 1000

        return result

    # ── Cancel ──

    async def cancel(self, session_key: str) -> bool:
        """Cancel an active run."""
        cancel = self._cancel_events.get(session_key)
        if cancel:
            cancel.set()

        task = self._active_runs.get(session_key)
        if task and not task.done():
            task.cancel()
            return True
        return False

    async def _cancel_existing(self, session_key: str) -> None:
        """Cancel any existing run for this session."""
        cancel = self._cancel_events.get(session_key)
        if cancel:
            cancel.set()
        task = self._active_runs.pop(session_key, None)
        if task and not task.done():
            task.cancel()
            try:
                await asyncio.wait_for(asyncio.shield(task), timeout=2)
            except (asyncio.CancelledError, asyncio.TimeoutError):
                pass  # intentional:  suppressed

    # ── Query ──

    def is_active(self, session_key: str) -> bool:
        task = self._active_runs.get(session_key)
        return task is not None and not task.done()

    def active_count(self) -> int:
        return sum(1 for t in self._active_runs.values() if not t.done())

    def list_active(self) -> List[str]:
        return [k for k, t in self._active_runs.items() if not t.done()]

from caveman.gateway.agent_runner_depth import (  # noqa: F401,E402  # depth wiring
    ToolProgress,
    StreamEvent,
    StreamingAgentRunner,
)

__all__ = [
    "RunContext",
    "RunResult",
    "AgentRunner",
    "ToolProgress",
    "StreamEvent",
    "StreamingAgentRunner",
]

