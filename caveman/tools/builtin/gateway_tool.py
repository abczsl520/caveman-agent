"""Gateway tools — send messages and list gateways."""
from __future__ import annotations

from caveman.tools.registry import tool


@tool(
    name="gateway_send",
    description="Send a message via a gateway",
    params={
        "gateway": {"type": "string", "description": "Gateway name (discord, telegram)"},
        "channel_id": {"type": "string", "description": "Channel/chat ID"},
        "content": {"type": "string", "description": "Message content"},
    },
    required=["gateway", "channel_id", "content"],
)
async def gateway_send(
    gateway: str, channel_id: str, content: str, _context: dict | None = None,
) -> dict:
    """Send a message through a named gateway."""
    router = (_context or {}).get("gateway_router")
    if not router:
        return {"error": "gateway_router not available"}
    try:
        result = await router.send(gateway, channel_id, content)
    except (ValueError, RuntimeError) as e:
        return {"error": str(e)}
    return {"ok": True, "result": result}


@tool(
    name="gateway_list",
    description="List available gateways",
    params={},
    required=[],
)
async def gateway_list(_context: dict | None = None) -> list[dict]:
    """List all registered gateways and their status."""
    router = (_context or {}).get("gateway_router")
    if not router:
        return [{"error": "gateway_router not available"}]
    return router.list_gateways()
