"""GatewayServer task handling helpers."""
from __future__ import annotations

import logging
from typing import Any

import caveman.gateway.runner as runner_module

logger = logging.getLogger("caveman.gateway")


async def run_locked_gateway_task(server: Any, task: str, context: dict[str, Any], key: str, msg_ctx: Any) -> str:
    """Run a normalized gateway task under its per-session lock."""
    lock = server._get_lock(key)
    async with lock:
        session = await server._get_or_create_session(key)
        gw_name = context.get("gateway_name", "discord")
        channel_id = str(context.get("channel_id", ""))
        source_channel = {
            "gateway": gw_name,
            "channel_id": channel_id,
            "user_id": context.get("user_id"),
            "message_id": context.get("message_id"),
            "_progress_sent": 0,
        }
        session["task_count"] += 1
        logger.info("Task #%d [%s]: %s", session["task_count"], key, task[:100])
        auto_mode = bool(getattr(server, "_auto_patterns", runner_module._AUTO_PATTERNS).search(task))
        if auto_mode:
            session.setdefault("auto_rounds", 0)
            source_channel["_auto_continue"] = True
        try:
            result = await runner_module.run_single_task(
                task, session, gw_name, channel_id, source_channel,
                server.router, server.store, server._cfg(),
                attachments=context.get("attachments"),
            )
            await server._infra.emit_hook("agent:end", {
                "session_key": key,
                "task": task[:200],
                "result_length": len(result or ""),
            })
            if auto_mode:
                result = await server._auto_continue(result, session, gw_name, channel_id, source_channel)
            queued = server._queue_manager.drain(key)
            for qm in queued:
                logger.info("Processing queued message for %s: %s", key, qm.body[:80])
                source_channel["_progress_sent"] = 0
                await runner_module.run_single_task(
                    qm.body, session, gw_name, channel_id, source_channel,
                    server.router, server.store, server._cfg(),
                )
            return ""
        except runner_module.AgentTaskError as exc:
            logger.warning("Task aborted by agent error: %s", exc)
            return ""
        except Exception as exc:
            logger.exception("Task failed: %s", exc)
            return (
                f"⚠️ 任务执行链路异常：{type(exc).__name__}: {str(exc)[:300]}。"
                "已记录日志；这不是任务完成信号，请继续排查。"
            )
