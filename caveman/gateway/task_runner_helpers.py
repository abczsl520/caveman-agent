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
        tool_name = event.data.get("name", "")
        tool_args = str(event.data.get("input", event.data.get("arguments", "")))
    ctx.tool_call_count += 1

    # Stuck-loop detection
    stuck = ctx.check_stuck_loop(tool_name, tool_args)
    if stuck:
        kind, detail = stuck.split(":", 1)
        if kind == "exact_repeat":
            msg = f"⚠️ 检测到循环：{detail} 连续 {_STUCK_LOOP_THRESHOLD} 次相同调用，暂停任务。"
        else:
            msg = f"⚠️ 检测到模式循环：{detail} 重复 {_PATTERN_LOOP_REPEATS} 次，暂停任务。"
        logger.warning("Stuck loop detected: %s", stuck)
        ctx.shutdown_flag = True
        await ctx.send(f"{msg}进度已保存，发消息可继续。")
        return True  # break

    async def _heartbeat(name: str):
        await asyncio.sleep(15.0)
        from caveman.tools.display import tool_display
        # Track tool call counts for compact display
        ctx._hb_counts[name] = ctx._hb_counts.get(name, 0) + 1
        # Build compact status line with emojis: "⏳ 🧠 Recall ×3, 💻 Shell ×2"
        parts = []
        for k, v in ctx._hb_counts.items():
            emoji, label = tool_display(k)
            parts.append(f"{emoji}{label} ×{v}" if v > 1 else f"{emoji}{label}")
        iter_info = f" [{ctx.iteration}/{ctx.max_iterations}]" if ctx.max_iterations else ""
        text = f"⏳{iter_info} {', '.join(parts)}..."
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
    loop_obj = session.get("loop")
    if not loop_obj or not loop_obj.nudge:
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
