"""Progress tool — send status updates to the current conversation channel.

Simpler than gateway_send: auto-detects the source channel from context.
Also tracks whether progress was sent, so gateway can skip duplicate final response.
"""
from __future__ import annotations

from caveman.tools.registry import tool


@tool(
    name="progress",
    description=(
        "Send a progress update to the user in the current conversation. "
        "Use this to report what you're doing, what you found, or when you're done. "
        "IMPORTANT: Use this every 2-3 tool calls to keep the user informed. "
        "On Discord/Telegram: start with an emoji (🔍📌⚡✅❌🔧) and use **bold** for key terms."
    ),
    params={
        "message": {
            "type": "string",
            "description": "Progress message to send to the user",
        },
    },
    required=["message"],
)
async def progress_send(message: str, _context: dict | None = None) -> dict:
    """Send progress update to current channel."""
    ctx = _context or {}
    router = ctx.get("gateway_router")
    source = ctx.get("source_channel", {})

    if not router:
        return {"ok": True, "note": "No gateway router (CLI mode), progress logged only"}

    gw_name = source.get("gateway", "discord")
    channel_id = source.get("channel_id", "")

    if not channel_id:
        return {"ok": True, "note": "No channel_id in context, progress logged only"}

    try:
        await router.send(gw_name, channel_id, message)
        # Mark that progress was sent — gateway can check this to skip final response
        source["_progress_sent"] = source.get("_progress_sent", 0) + 1
        return {"ok": True, "sent_to": f"{gw_name}:{channel_id}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}
