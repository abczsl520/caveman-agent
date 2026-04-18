"""Checkpoint tools — save and restore agent conversation state."""
from __future__ import annotations

from caveman.tools.registry import tool


@tool(
    name="checkpoint_save",
    description="Save current conversation state",
    params={
        "session_id": {"type": "string", "description": "Session ID to save"},
        "metadata": {"type": "object", "description": "Optional metadata", "default": {}},
    },
    required=["session_id"],
)
async def checkpoint_save(
    session_id: str, metadata: dict | None = None, _context: dict | None = None,
) -> dict:
    """Save a checkpoint of the current conversation."""
    mgr = (_context or {}).get("checkpoint_manager")
    if not mgr:
        return {"error": "checkpoint_manager not available"}
    from caveman.agent.checkpoint import Checkpoint
    messages = (_context or {}).get("messages", [])
    cp = Checkpoint(session_id, messages, metadata)
    try:
        cp_id = await mgr.save(cp)
    except (ValueError, OSError) as e:
        return {"error": f"Failed to save checkpoint: {e}"}
    return {"ok": True, "checkpoint_id": cp_id}


@tool(
    name="checkpoint_restore",
    description="Restore a conversation from checkpoint",
    params={
        "session_id": {"type": "string", "description": "Session ID to restore"},
        "checkpoint_id": {"type": "string", "description": "Specific checkpoint ID (optional)"},
    },
    required=["session_id"],
)
async def checkpoint_restore(
    session_id: str, checkpoint_id: str = "", _context: dict | None = None,
) -> dict:
    """Restore a conversation from a checkpoint."""
    mgr = (_context or {}).get("checkpoint_manager")
    if not mgr:
        return {"error": "checkpoint_manager not available"}
    cp = await mgr.restore(session_id, checkpoint_id or None)
    if not cp:
        return {"error": f"No checkpoint found for session {session_id}"}
    return {"ok": True, "session_id": cp.session_id, "messages_count": len(cp.messages)}


@tool(
    name="checkpoint_list",
    description="List checkpoints",
    params={
        "session_id": {"type": "string", "description": "Filter by session ID (optional)"},
    },
    required=[],
)
async def checkpoint_list(session_id: str = "", _context: dict | None = None) -> list[dict]:
    """List available checkpoints."""
    mgr = (_context or {}).get("checkpoint_manager")
    if not mgr:
        return [{"error": "checkpoint_manager not available"}]
    return await mgr.list_checkpoints(session_id or None)
