"""Task runner — executes a single agent task with activity monitoring.

Implements:
- Activity-based idle detection (replaces hard timeout)
- Stuck-loop detection (same tool+args repeated N times)
- Graceful shutdown (flag-based, not CancelledError)
- Three-layer streaming (interim + final messages)
- Child task tracking (no zombie tasks)
"""
from __future__ import annotations
import asyncio
import logging
import time
from typing import Any

from caveman.gateway.router import GatewayRouter
from caveman.gateway.smart_buffer import _SmartBuffer
from caveman.gateway.task_runner_helpers import _activity_monitor, _handle_tool_call, _persist_result, _spawn_post_task_review

logger = logging.getLogger("caveman.gateway")

# Defaults (overridable via gateway config)
_DEFAULT_PROGRESS_INTERVAL = 60.0   # 1 min between progress indicators
_DEFAULT_IDLE_WARNING = 180.0       # 3 min idle → warning
_DEFAULT_IDLE_SHUTDOWN = 300.0      # 5 min idle → graceful shutdown
_DEFAULT_ABSOLUTE_MAX = 1800.0      # 30 min absolute safety net
_STUCK_LOOP_THRESHOLD = 5           # Same tool+args repeated N times → abort
_PATTERN_LOOP_WINDOW = 20           # Window for pattern-based loop detection
_PATTERN_LOOP_REPEATS = 5           # Pattern must repeat N times to trigger

_ENDINGS = ("✅", "完成", "Done", "done.", "以上", "结束", "？", "?", "吗？", "吗?")

def _resolve_timeouts(config: dict[str, Any] | None) -> dict[str, float]:
    """Read user-configurable timeouts from gateway config."""
    defaults = {
        "progress_interval": _DEFAULT_PROGRESS_INTERVAL,
        "idle_warning": _DEFAULT_IDLE_WARNING,
        "idle_shutdown": _DEFAULT_IDLE_SHUTDOWN,
        "absolute_max": _DEFAULT_ABSOLUTE_MAX,
    }
    if not config:
        return defaults
    timeouts = config.get("gateway", {}).get("timeouts", {})
    if not isinstance(timeouts, dict):
        return defaults
    for key in defaults:
        val = timeouts.get(key)
        if val is not None:
            try:
                defaults[key] = float(val)
            except (TypeError, ValueError):
                pass  # intentional: TypeError/ValueError suppressed
    return defaults

class _TaskContext:
    """Mutable state for a single task execution."""

    __slots__ = (
        "gw_name", "channel_id", "router", "timeouts",
        "tool_call_count", "shutdown_flag", "idle_warned",
        "last_event_time", "last_user_visible_time", "task_start_time",
        "recent_tool_calls", "child_tasks", "tool_heartbeat",
        "_hb_msg_id", "_hb_counts", "iteration", "max_iterations",
        "pressure_warned", "stuck_warnings",
    )

    def __init__(self, gw_name: str, channel_id: str, router: GatewayRouter,
                 timeouts: dict[str, float]):
        self.gw_name = gw_name
        self.channel_id = channel_id
        self.router = router
        self.timeouts = timeouts
        self.tool_call_count = 0
        self.shutdown_flag = False
        self.idle_warned = False
        now = asyncio.get_running_loop().time()
        self.last_event_time = now
        self.last_user_visible_time = now
        self.task_start_time = now
        self.recent_tool_calls: list[str] = []
        self.child_tasks: set[asyncio.Task] = set()
        self.tool_heartbeat: asyncio.Task | None = None
        self.iteration = 0
        self.max_iterations = 0
        self.pressure_warned = False
        self.stuck_warnings = 0
        self._hb_msg_id: int | None = None  # Discord message ID for heartbeat edits
        self._hb_counts: dict[str, int] = {}  # tool_name → count for heartbeat display

    def touch_activity(self) -> None:
        """Reset idle timer on any stream event."""
        self.last_event_time = asyncio.get_running_loop().time()
        self.idle_warned = False

    def check_stuck_loop(self, tool_name: str, tool_args: str) -> str | None:
        """Detect stuck loops. Returns description if stuck, None if OK.

        Two detection modes:
        1. Exact repeat: same tool+args N times in a row
        2. Pattern loop: a sequence of 2-4 calls repeating N times (e.g. read→edit→read→edit→read→edit)
        """
        sig = f"{tool_name}:{hash(tool_args)}"
        self.recent_tool_calls.append(sig)
        # Keep window for pattern detection
        if len(self.recent_tool_calls) > _PATTERN_LOOP_WINDOW:
            self.recent_tool_calls.pop(0)
        calls = self.recent_tool_calls

        # Mode 1: exact repeat
        if len(calls) >= _STUCK_LOOP_THRESHOLD:
            if len(set(calls[-_STUCK_LOOP_THRESHOLD:])) == 1:
                return f"exact_repeat:{tool_name}"

        # Mode 2: pattern loop (detect repeating subsequences of length 2-4)
        if len(calls) >= 6:
            # Extract just tool names for pattern matching
            names = [c.split(":")[0] for c in calls]
            for pat_len in (2, 3, 4):
                if len(names) >= pat_len * _PATTERN_LOOP_REPEATS:
                    tail = names[-(pat_len * _PATTERN_LOOP_REPEATS):]
                    pattern = tail[:pat_len]
                    if all(tail[i] == pattern[i % pat_len] for i in range(len(tail))):
                        # Only flag if pattern has 2+ distinct tools (single-tool repeat is handled by exact_repeat)
                        if len(set(pattern)) >= 2:
                            return f"pattern_loop:{'→'.join(pattern)}"
        return None

    def spawn_task(self, coro, *, name: str, critical: bool = False) -> asyncio.Task:
        """Create a tracked asyncio task with exception logging."""
        ctx = self

        async def _wrapper():
            try:
                await coro
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
            except Exception as e:
                logger.error("Child task '%s' crashed: %s", name, e, exc_info=True)
                if critical:
                    ctx.shutdown_flag = True
                    logger.error("Critical task '%s' died — triggering shutdown", name)

        task = asyncio.create_task(_wrapper())
        self.child_tasks.add(task)
        task.add_done_callback(self.child_tasks.discard)
        return task

    def cancel_all(self) -> None:
        """Cancel all child tasks."""
        for t in self.child_tasks:
            if not t.done():
                t.cancel()
        self.child_tasks.clear()

    async def send(self, message: str) -> dict | None:
        """Send a message to the channel, swallowing non-critical errors."""
        try:
            return await self.router.send(self.gw_name, self.channel_id, message)
        except Exception as e:
            logger.debug("Non-critical send error: %s", e)


async def run_single_task(
    task: str, session: dict, gw_name: str, channel_id: str,
    source_channel: dict, router: GatewayRouter, store: Any,
    config: dict[str, Any] | None = None,
    attachments: list[dict[str, str]] | None = None,
) -> str:
    """Execute a single task and return the result text."""
    loop = session["loop"]
    timeouts = _resolve_timeouts(config)

    # Reset iteration budget for each new task
    loop.budget.reset()
    loop.tool_registry.set_context("source_channel", source_channel)
    loop.tool_registry.set_context("gateway_router", router)
    store.append_turn(session["meta"].session_id, "user", task)

    ctx = _TaskContext(gw_name, channel_id, router, timeouts)
    session["_task_ctx"] = ctx  # For interrupt support
    buf = _SmartBuffer(router, gw_name, channel_id)
    final_text = ""

    ctx.spawn_task(_activity_monitor(ctx), name="activity_monitor", critical=True)

    # Observability: log system prompt health before LLM call
    prompt_len = len(getattr(loop, '_system_prompt_cache', '') or '')
    surface = getattr(loop, 'surface', 'unknown')
    logger.info("Task start: surface=%s, prompt=%d chars, turn=%d",
                surface, prompt_len, getattr(loop, '_turn_number', 0))
    if prompt_len < 100:
        logger.error("🚨 System prompt critically short (%d chars)! Session may have lost its prompt.", prompt_len)

    try:
        async for event in loop.run_stream(task, attachments=attachments):
            if ctx.shutdown_flag:
                logger.info("Graceful shutdown: stopping stream processing")
                break

            ctx.touch_activity()

            if event.type == "token":
                await buf.add(str(event.data))

            elif event.type == "tool_call":
                if await _handle_tool_call(event, ctx, buf):
                    break

            elif event.type == "tool_result":
                ctx.touch_activity()
                if ctx.tool_heartbeat and not ctx.tool_heartbeat.done():
                    ctx.tool_heartbeat.cancel()
                    ctx.child_tasks.discard(ctx.tool_heartbeat)
                    ctx.tool_heartbeat = None
                # Stream long tool results to user (>500 chars)
                data = event.data or {}
                tool_name = data.get("tool_name", "")
                output = str(data.get("output", ""))
                if len(output) > 500 and tool_name in ("bash", "file_read", "session_search"):
                    preview = output[:300].rstrip()
                    await ctx.send(f"📋 `{tool_name}` 结果预览:\n```\n{preview}\n```")

            elif event.type == "error":
                await buf.flush()
                await ctx.send(f"⚠️ {str(event.data)[:500]}")

            elif event.type == "iteration_start":
                data = event.data or {}
                ctx.iteration = data.get("iteration", 0)
                ctx.max_iterations = data.get("max", 0)

            elif event.type == "context_pressure":
                data = event.data or {}
                util = data.get("utilization", 0)
                if util >= 0.9 and not ctx.pressure_warned:
                    ctx.pressure_warned = True
                    pct = int(util * 100)
                    await ctx.send(f"⚠️ 上下文已用 {pct}%，即将压缩。长对话建议开新 session。")

            elif event.type == "done":
                final_text = str(event.data) if event.data else ""
                await buf.flush()

        buf.cancel()
        ctx.cancel_all()
        session.pop("_task_ctx", None)  # Clear interrupt reference
        # Clean up heartbeat status message
        if ctx._hb_msg_id:
            try:
                await ctx.router.edit(ctx.gw_name, ctx.channel_id, ctx._hb_msg_id,
                                      f"✅ 完成 ({ctx.tool_call_count} 个工具调用)")
            except Exception as exc:
                logger.debug("unknown: suppressed %s", exc)
        _persist_result(buf, final_text, session, store)

        # Background review: extract memories from this task
        _spawn_post_task_review(session, ctx.tool_call_count)

        # Send completion marker if needed
        progress_count = source_channel.get("_progress_sent", 0)
        if buf.sent_any or progress_count > 0:
            last_sent = (buf._sent_text or "").strip()
            if last_sent and not any(last_sent.rstrip().endswith(e) for e in _ENDINGS):
                await ctx.send("✅")
            return final_text or ""
        return final_text or "Done."

    except Exception:
        try:
            await buf.flush()  # Send any buffered text before dying
        except Exception as exc:
            logger.debug("unknown: suppressed %s", exc)
        buf.cancel()
        ctx.cancel_all()
        raise
