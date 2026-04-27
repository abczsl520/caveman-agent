"""Task runner helper functions.

Internal helpers for task execution: activity monitoring, tool call handling,
result persistence, and post-task review spawning.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from pathlib import Path
from typing import Any, TYPE_CHECKING

if TYPE_CHECKING:
    from caveman.gateway.task_runner import _TaskContext

from caveman.aio import aio_write_text, aio_mkdir
from caveman.paths import CAVEMAN_HOME

_STUCK_LOOP_THRESHOLD = 5
_PATTERN_LOOP_REPEATS = 5


logger = logging.getLogger(__name__)

async def _activity_monitor(ctx: _TaskContext) -> None:
    """Activity-based idle detection + progress indicator."""
    while not ctx.shutdown_flag:
        await asyncio.sleep(min(15, ctx.timeouts['progress_interval'] / 4))
        now = asyncio.get_running_loop().time()
        idle_secs = now - ctx.last_event_time
        total_secs = now - ctx.task_start_time
        visible_gap = now - ctx.last_user_visible_time

        # Absolute safety net
        if total_secs >= ctx.timeouts['absolute_max']:
            logger.warning("Absolute timeout (%.0fs), graceful shutdown", total_secs)
            ctx.shutdown_flag = True
            await ctx.send(
                f"⏸️ 任务运行 {int(total_secs/60)} 分钟，已达到最长运行时间。"
                f"已执行 {ctx.tool_call_count} 个工具调用；任务未判定完成，进度已保存。发消息可继续。"
            )
            return

        # Idle shutdown
        if idle_secs >= ctx.timeouts['idle_shutdown']:
            logger.warning("Idle timeout (%.0fs), graceful shutdown", idle_secs)
            ctx.shutdown_flag = True
            await ctx.send(
                f"⏸️ {int(idle_secs/60)} 分钟无新进展，暂停任务。"
                f"已执行 {ctx.tool_call_count} 个工具调用；任务未判定完成，进度已保存。发消息可继续。"
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
            ctx.last_user_visible_time = asyncio.get_running_loop().time()

async def _handle_tool_call(event, ctx: _TaskContext, buf: _SmartBuffer) -> bool:
    """Handle a tool_call event. Returns True if should break the stream."""
    buf_before = len(buf._buf)
    interim = await buf.flush_interim()
    logger.info("tool_call boundary: buf_before=%d interim_len=%d sent_any=%s",
                buf_before, len(interim), buf.sent_any)
    if buf.sent_any:
        ctx.last_user_visible_time = asyncio.get_running_loop().time()

    tool_name = ""
    tool_args = ""
    if isinstance(event.data, dict):
        name = event.data.get("name", "")
        raw_args = event.data.get("input", event.data.get("arguments", ""))
        args = raw_args if isinstance(raw_args, dict) else {}
        tool_name = name
        tool_args = str(raw_args)
    ctx.tool_call_count += 1

    # Optional explicit tool-call budget. By default there is no arbitrary
    # per-task tool-count cap: long flywheel/audit tasks should be governed by
    # progress visibility, timeouts, permissions, and stuck-loop detection.
    # If configured, budget exhaustion pauses as incomplete/continuable work.
    if ctx.max_tool_calls is not None and ctx.tool_call_count >= ctx.max_tool_calls:
        ctx.shutdown_flag = True
        await ctx.send(
            f"⏸️ 单次任务已执行 {ctx.tool_call_count} 个工具调用，达到你配置的工具预算 "
            f"({ctx.max_tool_calls})。任务未判定完成，进度已保存，发消息可继续或提高 `gateway.limits.max_tool_calls`。"
        )
        return True

    # Stuck-loop detection
    stuck = ctx.check_stuck_loop(tool_name, tool_args)
    if stuck:
        ctx.stuck_warnings += 1
        kind, detail = stuck.split(":", 1)
        if kind == "exact_repeat":
            msg = f"⚠️ 检测到循环：{detail} 连续 {_STUCK_LOOP_THRESHOLD} 次相同调用"
        else:
            msg = f"⚠️ 检测到模式循环：{detail} 重复 {_PATTERN_LOOP_REPEATS} 次"
        logger.warning("Stuck loop detected (strike %d): %s", ctx.stuck_warnings, stuck)
        if ctx.stuck_warnings >= 2:
            # Second strike — hard shutdown
            ctx.shutdown_flag = True
            await ctx.send(f"{msg}，二次触发，暂停任务。进度已保存，发消息可继续。")
            return True  # break
        else:
            # First strike — warn and clear history so agent can self-correct
            ctx.recent_tool_calls.clear()
            await ctx.send(f"{msg}。请换个思路继续。")
            return False  # let agent continue

    async def _send_or_edit_heartbeat(name: str) -> None:
        """Send or edit a bounded visible tool heartbeat."""
        from caveman.tools.display import tool_display

        ctx._hb_counts[name] = ctx._hb_counts.get(name, 0) + 1
        total = sum(ctx._hb_counts.values())
        recent = list(ctx._hb_counts.items())[-5:]
        parts = []
        for k, v in recent:
            emoji, label = tool_display(k)
            safe_label = label if label != k else "tool"
            parts.append(f"{emoji}{safe_label} ×{v}" if v > 1 else f"{emoji}{safe_label}")
        if len(ctx._hb_counts) > len(recent):
            parts.insert(0, "工具调用持续进行中")
        iter_info = f" [{ctx.iteration}/{ctx.max_iterations}]" if ctx.max_iterations else ""
        elapsed = int(asyncio.get_running_loop().time() - ctx.task_start_time)
        mins, secs = divmod(elapsed, 60)
        text = f"⏳{iter_info} {', '.join(parts)} ({f'{mins}m{secs:02d}s' if mins else f'{secs}s'})..."
        try:
            if ctx._hb_msg_id:
                await ctx.router.edit(ctx.gw_name, ctx.channel_id, ctx._hb_msg_id, text)
            else:
                result = await ctx.router.send(ctx.gw_name, ctx.channel_id, text)
                if isinstance(result, dict) and result.get("message_id"):
                    ctx._hb_msg_id = result["message_id"]
                elif isinstance(result, (str, int)) and result:
                    ctx._hb_msg_id = result
        except Exception as e:
            logger.debug("Heartbeat send/edit failed: %s", e)

    async def _heartbeat(name: str):
        """Periodic heartbeat during tool execution.

        Loops every 15s to:
        1. touch_activity() - prevent idle_shutdown from misfiring
        2. Send/edit a status message so the user sees progress

        Before this fix, _heartbeat ran only ONCE (sleep 15s, send, exit).
        Long tools like coding_agent (15min) would trigger idle_shutdown
        (5min) because touch_activity was never called during execution.
        """
        while True:
            await asyncio.sleep(15.0)
            # Critical: keep idle detector alive during long tool execution
            ctx.touch_activity()
            await _send_or_edit_heartbeat(name)

    await _send_or_edit_heartbeat(tool_name)
    ctx.tool_heartbeat = ctx.spawn_task(lambda: _heartbeat(tool_name), name=f"heartbeat:{tool_name}")
    return False

def _persist_result(buf: _SmartBuffer, final_text: str, session: dict, store: Any, *, continuation_task: bool = False) -> None:
    """Save result to session store and update metadata."""
    meta = session.get("meta")
    loop = session.get("loop") or session.get("agent_loop")

    save_text = buf._sent_text.strip() or final_text or buf._full_text
    if continuation_task:
        from caveman.agent.output_validator import suppress_continuation_terminality
        save_text = suppress_continuation_terminality(save_text, task="继续飞轮", surface=getattr(loop, "surface", "cli"))
    if meta is None or loop is None:
        logger.debug("Persisting result without session meta")
        try:
            store.append_turn("unknown", "assistant", save_text[:16000])
        except Exception as exc:
            logger.debug("Fallback result persistence failed: %s", exc)
        return
    # Tool call count is metadata, not message content.
    # Injecting it into the text pollutes conversation history and causes
    # cumulative snowball on session restore (the count keeps growing).
    # Store it in session meta instead.
    tool_count = getattr(loop, '_tool_call_count', 0)
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

    # Session size warning — alert if getting large
    if meta.total_tokens > 500_000 and meta.turn_count % 10 == 0:
        logger.warning(
            "Session %s is large: %d tokens, %d turns, $%.4f",
            meta.session_id, meta.total_tokens, meta.turn_count, meta.total_cost_usd,
        )

    # Persist loop snapshot for reliable restore
    if hasattr(loop, 'snapshot'):
        try:
            snap = loop.snapshot()
            store.save_snapshot(meta.session_id, snap)
        except Exception as e:
            logger.warning("Snapshot save failed: %s", e)

_REVIEW_MIN_TOOL_CALLS = 3  # Only review tasks with meaningful work

def _spawn_post_task_review(session: dict, tool_call_count: int) -> None:
    """Spawn a background task to extract memories from the completed task.

    Only runs if the task had enough tool calls to be worth reviewing.
    Uses the existing Nudge engine (no extra LLM call).
    """
    if tool_call_count < _REVIEW_MIN_TOOL_CALLS:
        return
    loop_obj = session.get("loop") or session.get("agent_loop")
    if not loop_obj or not getattr(loop_obj, "nudge", None):
        return
    nudge = loop_obj.nudge
    trajectory = loop_obj.trajectory_recorder
    task_desc = loop_obj.nudge_task_ref

    async def _review():
        try:
            turns = trajectory.to_sharegpt()[-20:]
            if not turns:
                return
            created = await nudge.run(turns, task=task_desc)
            if created:
                logger.info("Post-task review: extracted %d memories", len(created))
        except Exception as e:
            logger.debug("Post-task review failed: %s", e)

    try:
        asyncio.get_running_loop().create_task(_review(), name="post_task_review")
    except RuntimeError:
        pass  # No running loop (shouldn't happen in gateway)
