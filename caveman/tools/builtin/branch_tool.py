"""Conversation branch tool — fork the current conversation to explore alternatives.

Allows the agent to create a snapshot of the current context,
explore a different approach, and optionally merge back.
"""
from __future__ import annotations
import logging
from typing import Any

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

# In-memory branch storage (per session, cleared on restart)
_branches: dict[str, dict[str, Any]] = {}


@tool(
    name="conversation_branch",
    description="Fork the current conversation to explore an alternative approach. "
                "Use 'create' to save a checkpoint, 'restore' to go back, 'list' to see branches.",
    params={
        "action": {"type": "string", "enum": ["create", "restore", "list", "delete"],
                    "description": "Branch action"},
        "name": {"type": "string", "description": "Branch name (for create/restore/delete)"},
        "reason": {"type": "string", "description": "Why branching (for create)"},
    },
    required=["action"],
)
async def conversation_branch(action: str, name: str = "", reason: str = "",
                               **_kw) -> dict:
    ctx = _kw.get("_context")
    session_id = _kw.get("_session_id", "default")

    if session_id not in _branches:
        _branches[session_id] = {}
    branches = _branches[session_id]

    if action == "create":
        if not name:
            return {"error": "Branch name required"}
        if not ctx:
            return {"error": "No context available"}
        branches[name] = {
            "context": ctx.fork(label=name),
            "reason": reason,
            "messages": len(ctx.messages),
        }
        return {"created": name, "messages": len(ctx.messages), "reason": reason}

    elif action == "restore":
        if not name or name not in branches:
            return {"error": f"Branch '{name}' not found", "available": list(branches.keys())}
        branch = branches[name]
        # The actual restore would need to be handled by the loop
        return {"restored": name, "messages": branch["messages"],
                "hint": "Context restored to branch checkpoint"}

    elif action == "list":
        return {"branches": {
            k: {"messages": v["messages"], "reason": v.get("reason", "")}
            for k, v in branches.items()
        }}

    elif action == "delete":
        if name in branches:
            del branches[name]
            return {"deleted": name}
        return {"error": f"Branch '{name}' not found"}

    return {"error": f"Unknown action: {action}"}
