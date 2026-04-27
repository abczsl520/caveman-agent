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
from typing import Any


from caveman.agent.stream import is_result_event_type
from caveman.agent.output_validator import suppress_continuation_terminality, is_continuation_task
from caveman.gateway.router import GatewayRouter
from caveman.gateway.smart_buffer import _SmartBuffer
from caveman.gateway.task_runner_helpers import _activity_monitor, _handle_tool_call, _persist_result, _spawn_post_task_review

logger = logging.getLogger("caveman.gateway")


class AgentTaskError(RuntimeError):
    """Agent task failed after the error was already reported to the user."""


# Defaults (overridable via gateway config)
_DEFAULT_PROGRESS_INTERVAL = 60.0   # 1 min between progress indicators
_DEFAULT_IDLE_WARNING = 180.0       # 3 min idle → warning
_DEFAULT_IDLE_SHUTDOWN = 900.0      # 15 min idle → graceful shutdown
_DEFAULT_ABSOLUTE_MAX = 1800.0      # 30 min absolute safety net
_STUCK_LOOP_THRESHOLD = 5           # Same tool+args repeated N times → abort
_PATTERN_LOOP_WINDOW = 20           # Window for pattern-based loop detection
_PATTERN_LOOP_REPEATS = 5           # Pattern must repeat N times to trigger

def _resolve_timeouts(config: dict[str, Any] | None) -> dict[str, float | int | None]:
    """Read user-configurable runtime policy from gateway config.

    Long-compounding work must not be cut off by hidden tool-count caps. The
    time-based guards remain defaults, while tool-call budget is opt-in via
    `gateway.limits.max_tool_calls`.
    """
    defaults: dict[str, float | int | None] = {
        "progress_interval": _DEFAULT_PROGRESS_INTERVAL,
        "idle_warning": _DEFAULT_IDLE_WARNING,
        "idle_shutdown": _DEFAULT_IDLE_SHUTDOWN,
        "absolute_max": _DEFAULT_ABSOLUTE_MAX,
        "max_tool_calls": None,
    }
    if not config:
        return defaults
    gateway = config.get("gateway", {})
    timeouts = gateway.get("timeouts", {}) if isinstance(gateway, dict) else {}
    if isinstance(timeouts, dict):
        for key in ("progress_interval", "idle_warning", "idle_shutdown", "absolute_max"):
            val = timeouts.get(key)
            if val is not None:
                try:
                    defaults[key] = float(val)
                except (TypeError, ValueError):
                    pass  # intentional: TypeError/ValueError suppressed

    limits = gateway.get("limits", {}) if isinstance(gateway, dict) else {}
    if isinstance(limits, dict):
        val = limits.get("max_tool_calls")
        if val is not None:
            try:
                parsed = int(val)
                if parsed > 0:
                    defaults["max_tool_calls"] = parsed
            except (TypeError, ValueError):
                pass  # invalid optional limit is ignored here; config validator reports it
    return defaults

class _TaskContext:
    """Mutable state for a single task execution."""

    __slots__ = (
        "gw_name", "channel_id", "router", "timeouts",
        "tool_call_count", "max_tool_calls", "shutdown_flag", "idle_warned",
        "last_event_time", "last_user_visible_time", "task_start_time",
        "recent_tool_calls", "child_tasks", "tool_heartbeat",
        "_hb_msg_id", "_hb_counts", "iteration", "max_iterations",
        "pressure_warned", "stuck_warnings", "visible_message_count",
    )

    def __init__(self, gw_name: str, channel_id: str, router: GatewayRouter,
                 timeouts: dict[str, float]):
        self.gw_name = gw_name
        self.channel_id = channel_id
        self.router = router
        self.timeouts = timeouts
        self.tool_call_count = 0
        limit = timeouts.get("max_tool_calls")
        self.max_tool_calls = int(limit) if isinstance(limit, (int, float)) and limit > 0 else None
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
        self.visible_message_count = 0
        self._hb_msg_id: int | None = None  # Discord message ID for heartbeat edits
        self._hb_counts: dict[str, int] = {}  # tool_name → count for heartbeat display

    def touch_activity(self) -> None:
        """Reset idle timer on any stream event."""
        self.last_event_time = asyncio.get_running_loop().time()
        self.idle_warned = False

    # Tools that legitimately need repeated identical calls (polling patterns)
    _POLLING_TOOLS = frozenset({"process_output", "process_list", "acp_status"})

    def check_stuck_loop(self, tool_name: str, tool_args: str) -> str | None:
        """Detect stuck loops. Returns description if stuck, None if OK.

        Two detection modes:
        1. Exact repeat: same tool+args N times in a row
        2. Pattern loop: a sequence of 2-4 calls repeating N times (e.g. read→edit→read→edit→read→edit)

        Polling tools (process_output, acp_status) are exempt from exact-repeat
        detection since they legitimately need repeated identical calls to check
        on background processes. They still count for pattern-loop detection
        (e.g. process_output→file_read→process_output→file_read is suspicious).
        """
        sig = f"{tool_name}:{hash(tool_args)}"
        self.recent_tool_calls.append(sig)
        # Keep window for pattern detection
        if len(self.recent_tool_calls) > _PATTERN_LOOP_WINDOW:
            self.recent_tool_calls.pop(0)
        calls = self.recent_tool_calls

        # Mode 1: exact repeat (skip for polling tools)
        if tool_name not in self._POLLING_TOOLS:
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

    def spawn_task(self, coro_factory, *, name: str, critical: bool = False) -> asyncio.Task:
        """Create a tracked asyncio task with exception logging.

        Accepts either a coroutine object or a zero-arg coroutine factory. The
        factory form avoids "coroutine was never awaited" warnings if the task is
        cancelled before its first scheduling step.
        """
        ctx = self

        async def _wrapper():
            try:
                coro = coro_factory() if callable(coro_factory) else coro_factory
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

    async def cancel_all(self) -> None:
        """Cancel all child tasks and wait for cancellation to settle."""
        tasks = [t for t in self.child_tasks if not t.done()]
        for t in tasks:
            t.cancel()
        if tasks:
            await asyncio.gather(*tasks, return_exceptions=True)
        self.child_tasks.clear()

    async def send(self, message: str) -> dict | None:
        """Send a message to the channel, swallowing non-critical errors."""
        try:
            result = await self.router.send(self.gw_name, self.channel_id, message)
            self.visible_message_count += 1
            self.last_user_visible_time = asyncio.get_running_loop().time()
            return result
        except Exception as e:
            logger.debug("Non-critical send error: %s", e)


async def run_single_task(
    task: str, session: dict, gw_name: str, channel_id: str,
    source_channel: dict, router: GatewayRouter, store: Any,
    config: dict[str, Any] | None = None,
    attachments: list[dict[str, str]] | None = None,
) -> str:
    """Execute a single task and return the result text."""
    loop = session.get("loop") or session.get("agent_loop")
    if loop is None:
        raise AgentTaskError("Gateway session missing agent loop")
    timeouts = _resolve_timeouts(config)

    # Reset iteration budget for each new task when running a real AgentLoop.
    # Older restored sessions and lightweight test doubles may not expose the
    # newer budget/tool_registry attributes; don't turn that into a user-visible
    # generic gateway failure.
    budget = getattr(loop, "budget", None)
    if budget is not None and hasattr(budget, "reset"):
        budget.reset()
    tool_registry = getattr(loop, "tool_registry", None)
    if tool_registry is not None and hasattr(tool_registry, "set_context"):
        tool_registry.set_context("source_channel", source_channel)
        tool_registry.set_context("gateway_router", router)
    meta = session.get("meta")
    session_id = getattr(meta, "session_id", None)
    if session_id is not None:
        store.append_turn(session_id, "user", task)

    ctx = _TaskContext(gw_name, channel_id, router, timeouts)
    session["_task_ctx"] = ctx  # For interrupt support
    continuation_task = bool(source_channel.get("_auto_continue")) or is_continuation_task(task)
    # In automatic continuation mode the model's result paragraph is control-plane
    # input for the next round, not a user-visible final answer. Progress/tool
    # messages remain visible through progress/router sends; suppressing the final
    # paragraph avoids the repeated "done/收尾" illusion that made the flywheel look
    # stopped after every round.
    suppress_final_text = continuation_task
    buf = _SmartBuffer(router, gw_name, channel_id, send_enabled=not suppress_final_text)
    final_text = ""

    ctx.spawn_task(lambda: _activity_monitor(ctx), name="activity_monitor", critical=True)

    # Observability: log system prompt health before LLM call
    prompt_len = len(getattr(loop, '_system_prompt_cache', '') or '')
    surface = getattr(loop, 'surface', 'unknown')
    logger.info("Task start: surface=%s, prompt=%d chars, turn=%d",
                surface, prompt_len, getattr(loop, '_turn_number', 0))
    if prompt_len < 100:
        logger.error("🚨 System prompt critically short (%d chars)! Session may have lost its prompt.", prompt_len)

    try:
        stream = loop.run_stream(task, attachments=attachments) if attachments else loop.run_stream(task)
        async for event in stream:
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
                message = str(event.data)[:500]
                await ctx.send(f"⚠️ {message}")
                raise AgentTaskError(message)

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

            elif is_result_event_type(event.type):
                final_text = str(event.data) if event.data else ""
                if continuation_task:
                    final_text = suppress_continuation_terminality(
                        final_text, task=task, surface=gw_name
                    )
                await buf.flush(send=not suppress_final_text)
                if (
                    final_text
                    and not suppress_final_text
                    and not getattr(buf, "_sent_text", "").strip()
                ):
                    await ctx.send(final_text)
                elif suppress_final_text:
                    # Suppressing terminal-looking final paragraphs is necessary
                    # for auto flywheel, but a totally quiet round feels like the
                    # system stopped. If no progress/tool heartbeat/user-visible
                    # output happened in this round, emit one explicit non-final
                    # pulse so the control loop remains observable.
                    progress_count = int(source_channel.get("_progress_sent", 0) or 0)
                    if progress_count <= 0 and ctx.visible_message_count <= 0 and not ctx._hb_msg_id:
                        await ctx.send(
                            "🔄 自动续轮仍在推进；本轮没有产生可展示的最终文本，"
                            "不代表已完成或停止，继续进入下一步排查。"
                        )
                # Terminal completion markers are disabled by default. Do not
                # send a marker-only message; it causes gateway/flywheel to
                # treat partially verified work as complete.

        buf.cancel()
        await ctx.cancel_all()
        session.pop("_task_ctx", None)  # Clear interrupt reference
        # Clean up heartbeat status message. For continuation workflows, avoid
        # editing the visible heartbeat into a stop-looking terminal status;
        # the next progress/auto-round message is the source of truth.
        if ctx._hb_msg_id:
            source_channel["_hb_msg_id"] = ctx._hb_msg_id
        if ctx._hb_msg_id and not continuation_task:
            try:
                await ctx.router.edit(ctx.gw_name, ctx.channel_id, ctx._hb_msg_id,
                                      f"📌 响应流已停止（{ctx.tool_call_count} 个工具调用；不代表任务已验证完成，结果以最终消息/后续继续为准）")
            except Exception as exc:
                logger.debug("unknown: suppressed %s", exc)
        _persist_result(buf, final_text, session, store, continuation_task=continuation_task)

        # Background review: extract memories from this task
        _spawn_post_task_review(session, ctx.tool_call_count)

        # Do not fabricate "Done." when the agent produced no final text.
        # Returning an empty string lets callers distinguish "ended" from
        # "verified complete" and prevents flywheel/gateway false positives.
        # For auto/continuation work, return the sanitized non-terminal summary;
        # the raw final paragraph must not be fed back into the next auto round.
        return final_text or ""

    except Exception:
        try:
            await buf.flush()  # Send any buffered text before dying
        except Exception as exc:
            logger.debug("unknown: suppressed %s", exc)
        buf.cancel()
        await ctx.cancel_all()
        session.pop("_task_ctx", None)
        raise
