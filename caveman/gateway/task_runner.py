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

logger = logging.getLogger("caveman.gateway")

# Defaults (overridable via gateway config)
_DEFAULT_PROGRESS_INTERVAL = 60.0   # 1 min between progress indicators
_DEFAULT_IDLE_WARNING = 180.0       # 3 min idle → warning
_DEFAULT_IDLE_SHUTDOWN = 300.0      # 5 min idle → graceful shutdown
_DEFAULT_ABSOLUTE_MAX = 1800.0      # 30 min absolute safety net
_STUCK_LOOP_THRESHOLD = 5           # Same tool+args repeated N times → abort

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
                pass
    return defaults


class _TaskContext:
    """Mutable state for a single task execution."""

    __slots__ = (
        "gw_name", "channel_id", "router", "timeouts",
        "tool_call_count", "shutdown_flag", "idle_warned",
        "last_event_time", "last_user_visible_time", "task_start_time",
        "recent_tool_calls", "child_tasks", "tool_heartbeat",
        "_hb_msg_id", "_hb_counts",
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
        now = asyncio.get_event_loop().time()
        self.last_event_time = now
        self.last_user_visible_time = now
        self.task_start_time = now
        self.recent_tool_calls: list[str] = []
        self.child_tasks: set[asyncio.Task] = set()
        self.tool_heartbeat: asyncio.Task | None = None
        self._hb_msg_id: int | None = None  # Discord message ID for heartbeat edits
        self._hb_counts: dict[str, int] = {}  # tool_name → count for heartbeat display

    def touch_activity(self) -> None:
        """Reset idle timer on any stream event."""
        self.last_event_time = asyncio.get_event_loop().time()
        self.idle_warned = False

    def check_stuck_loop(self, tool_name: str, tool_args: str) -> bool:
        """Detect if the same tool+args is being called repeatedly."""
        sig = f"{tool_name}:{hash(tool_args)}"
        self.recent_tool_calls.append(sig)
        if len(self.recent_tool_calls) > _STUCK_LOOP_THRESHOLD:
            self.recent_tool_calls.pop(0)
        if len(self.recent_tool_calls) >= _STUCK_LOOP_THRESHOLD:
            return len(set(self.recent_tool_calls)) == 1
        return False

    def spawn_task(self, coro, *, name: str, critical: bool = False) -> asyncio.Task:
        """Create a tracked asyncio task with exception logging."""
        ctx = self

        async def _wrapper():
            try:
                await coro
            except asyncio.CancelledError:
                pass
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


async def _activity_monitor(ctx: _TaskContext) -> None:
    """Activity-based idle detection + progress indicator."""
    while not ctx.shutdown_flag:
        await asyncio.sleep(min(15, ctx.timeouts['progress_interval'] / 4))
        now = asyncio.get_event_loop().time()
        idle_secs = now - ctx.last_event_time
        total_secs = now - ctx.task_start_time
        visible_gap = now - ctx.last_user_visible_time

        # Absolute safety net
        if total_secs >= ctx.timeouts['absolute_max']:
            logger.warning("Absolute timeout (%.0fs), graceful shutdown", total_secs)
            ctx.shutdown_flag = True
            await ctx.send(
                f"⏸️ 任务运行 {int(total_secs/60)} 分钟，已达安全上限。"
                f"已完成 {ctx.tool_call_count} 个工具调用，进度已保存。发消息可继续。"
            )
            return

        # Idle shutdown
        if idle_secs >= ctx.timeouts['idle_shutdown']:
            logger.warning("Idle timeout (%.0fs), graceful shutdown", idle_secs)
            ctx.shutdown_flag = True
            await ctx.send(
                f"⏸️ {int(idle_secs/60)} 分钟无新进展，暂停任务。"
                f"已完成 {ctx.tool_call_count} 个工具调用，进度已保存。发消息可继续。"
            )
            return

        # Idle warning
        if idle_secs >= ctx.timeouts['idle_warning'] and not ctx.idle_warned:
            ctx.idle_warned = True
            await ctx.send(f"⚠️ {int(idle_secs/60)} 分钟无新进展，可能卡住了...")
            ctx.last_user_visible_time = now
            continue

        # Progress indicator
        if visible_gap >= ctx.timeouts['progress_interval'] - 5 and ctx.tool_call_count > 0:
            mins = int(total_secs / 60)
            await ctx.send(f"🔄 分析中... ({ctx.tool_call_count} 个工具调用, {mins}分钟)")
            ctx.last_user_visible_time = asyncio.get_event_loop().time()


async def _handle_tool_call(event, ctx: _TaskContext, buf: _SmartBuffer) -> bool:
    """Handle a tool_call event. Returns True if should break the stream."""
    buf_before = len(buf._buf)
    interim = await buf.flush_interim()
    logger.info("tool_call boundary: buf_before=%d interim_len=%d sent_any=%s",
                buf_before, len(interim), buf.sent_any)
    if buf.sent_any:
        ctx.last_user_visible_time = asyncio.get_event_loop().time()

    tool_name = ""
    tool_args = ""
    if isinstance(event.data, dict):
        tool_name = event.data.get("name", "")
        tool_args = str(event.data.get("input", event.data.get("arguments", "")))
    ctx.tool_call_count += 1

    # Stuck-loop detection
    if ctx.check_stuck_loop(tool_name, tool_args):
        logger.warning("Stuck loop detected: %s called %d times with same args",
                       tool_name, _STUCK_LOOP_THRESHOLD)
        ctx.shutdown_flag = True
        await ctx.send(
            f"⚠️ 检测到循环：{tool_name} 连续 {_STUCK_LOOP_THRESHOLD} 次相同调用，暂停任务。"
            f"进度已保存，发消息可继续。"
        )
        return True  # break

    async def _heartbeat(name: str):
        await asyncio.sleep(15.0)
        # Track tool call counts for compact display
        ctx._hb_counts[name] = ctx._hb_counts.get(name, 0) + 1
        # Build compact status line: "⏳ memory_search ×3, bash ×2"
        parts = [f"{k} ×{v}" if v > 1 else k for k, v in ctx._hb_counts.items()]
        text = f"⏳ {', '.join(parts)}..."
        try:
            if ctx._hb_msg_id:
                # Edit existing heartbeat message
                await ctx.router.edit(ctx.gw_name, ctx.channel_id, ctx._hb_msg_id, text)
            else:
                # Send new heartbeat message, save ID for future edits
                result = await ctx.router.send(ctx.gw_name, ctx.channel_id, text)
                if isinstance(result, dict) and result.get("message_id"):
                    ctx._hb_msg_id = result["message_id"]
        except Exception as e:
            logger.debug("Heartbeat send/edit failed: %s", e)

    ctx.tool_heartbeat = ctx.spawn_task(_heartbeat(tool_name), name=f"heartbeat:{tool_name}")
    return False


def _persist_result(buf: _SmartBuffer, final_text: str, session: dict, store: Any) -> None:
    """Save result to session store and update metadata."""
    meta = session["meta"]
    loop = session["loop"]

    save_text = buf._sent_text.strip() or final_text or buf._full_text
    # Include tool call summary for context restoration
    tool_count = getattr(loop, '_tool_call_count', 0)
    if tool_count > 0:
        save_text = f"[使用了 {tool_count} 个工具调用]\n{save_text}"
    store.append_turn(meta.session_id, "assistant", save_text[:16000])
    meta.turn_count += 1
    meta.last_active_at = time.time()

    try:
        usage = loop.provider.usage_stats
    except Exception:
        usage = None
    if isinstance(usage, dict):
        meta.total_tokens = usage.get('total_input_tokens', 0) + usage.get('total_output_tokens', 0)
        inp = usage.get('total_input_tokens', 0)
        out = usage.get('total_output_tokens', 0)
        try:
            from caveman.providers.model_metadata import get_model_info
            info = get_model_info(getattr(loop.provider, 'model', ''))
            meta.total_cost_usd = info.estimate_cost(inp, out)
        except Exception:
            meta.total_cost_usd = (inp * 3 + out * 15) / 1_000_000  # Sonnet fallback

    store.save_meta(meta)

    # Persist loop snapshot for reliable restore
    if hasattr(loop, 'snapshot'):
        try:
            snap = loop.snapshot()
            snap_path = store._session_dir(meta.session_id) / "loop_snapshot.json"
            import json
            tmp_path = snap_path.with_suffix('.tmp')
            tmp_path.write_text(json.dumps(snap))
            tmp_path.rename(snap_path)  # Atomic on POSIX
        except Exception as e:
            logger.warning("Snapshot save failed: %s", e)


async def run_single_task(
    task: str, session: dict, gw_name: str, channel_id: str,
    source_channel: dict, router: GatewayRouter, store: Any,
    config: dict[str, Any] | None = None,
) -> str:
    """Execute a single task and return the result text."""
    loop = session["loop"]
    timeouts = _resolve_timeouts(config)

    loop.tool_registry.set_context("source_channel", source_channel)
    loop.tool_registry.set_context("gateway_router", router)
    store.append_turn(session["meta"].session_id, "user", task)

    ctx = _TaskContext(gw_name, channel_id, router, timeouts)
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
        async for event in loop.run_stream(task):
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

            elif event.type == "error":
                await buf.flush()
                await ctx.send(f"⚠️ {str(event.data)[:500]}")

            elif event.type == "done":
                final_text = str(event.data) if event.data else ""
                await buf.flush()

        buf.cancel()
        ctx.cancel_all()
        # Clean up heartbeat status message
        if ctx._hb_msg_id:
            try:
                await ctx.router.edit(ctx.gw_name, ctx.channel_id, ctx._hb_msg_id,
                                      f"✅ 完成 ({ctx.tool_call_count} 个工具调用)")
            except Exception:
                pass
        _persist_result(buf, final_text, session, store)

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
        except Exception:
            pass
        buf.cancel()
        ctx.cancel_all()
        raise
