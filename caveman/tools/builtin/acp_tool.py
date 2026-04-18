"""ACP tools — send tasks to remote ACP agents."""
from __future__ import annotations

import logging

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="acp_send",
    description="Send a task to a remote ACP agent",
    params={
        "url": {"type": "string", "description": "Base URL of the ACP agent"},
        "message": {"type": "string", "description": "Message to send"},
        "wait": {"type": "boolean", "description": "Wait for completion", "default": True},
    },
    required=["url", "message"],
)
async def acp_send(url: str, message: str, wait: bool = True, **_kw) -> dict:
    """Send a task to a remote ACP agent."""
    from caveman.acp.client import ACPClient

    client = ACPClient(url)
    try:
        if wait:
            result = await client.send_task(message)
            return {"ok": True, **result}
        else:
            task_id = await client.send_task_async(message)
            return {"ok": True, "task_id": task_id}
    except Exception as e:
        logger.warning("acp_send failed: %s", e)
        return {"error": f"ACP send failed: {e}"}
    finally:
        await client.close()


@tool(
    name="acp_status",
    description="Check status of a remote ACP task",
    params={
        "url": {"type": "string", "description": "Base URL of the ACP agent"},
        "task_id": {"type": "string", "description": "Task ID to check"},
    },
    required=["url", "task_id"],
)
async def acp_status(url: str, task_id: str, **_kw) -> dict:
    """Check status of a remote ACP task."""
    from caveman.acp.client import ACPClient

    client = ACPClient(url)
    try:
        result = await client.get_task(task_id)
        return {"ok": True, **result}
    except Exception as e:
        logger.warning("acp_status failed: %s", e)
        return {"error": f"ACP status check failed: {e}"}
    finally:
        await client.close()


@tool(
    name="acp_cancel",
    description="Cancel a remote ACP task",
    params={
        "url": {"type": "string", "description": "Base URL of the ACP agent"},
        "task_id": {"type": "string", "description": "Task ID to cancel"},
    },
    required=["url", "task_id"],
)
async def acp_cancel(url: str, task_id: str, **_kw) -> dict:
    """Cancel a remote ACP task."""
    from caveman.acp.client import ACPClient

    client = ACPClient(url)
    try:
        result = await client.cancel_task(task_id)
        return {"ok": True, **result}
    except Exception as e:
        logger.warning("acp_cancel failed: %s", e)
        return {"error": f"ACP cancel failed: {e}"}
    finally:
        await client.close()
