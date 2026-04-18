"""Todo tool — persistent task list stored as JSON."""
from __future__ import annotations

import json
import logging
from datetime import datetime
from pathlib import Path
from uuid import uuid4

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_TODO_FILE = Path.home() / ".caveman" / "todos.json"


def _load() -> list[dict]:
    if _TODO_FILE.exists():
        try:
            return json.loads(_TODO_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt todos.json, starting fresh")
    return []


def _save(todos: list[dict]) -> None:
    _TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
    _TODO_FILE.write_text(json.dumps(todos, indent=2))


_VALID_PRIORITIES = {"low", "medium", "high"}


@tool(
    name="todo_add",
    description="Add a todo item",
    params={
        "title": {"type": "string", "description": "Todo title"},
        "priority": {"type": "string", "description": "Priority: low/medium/high", "default": "medium"},
    },
    required=["title"],
)
async def todo_add(title: str, priority: str = "medium") -> dict:
    """Add a new todo item."""
    if priority not in _VALID_PRIORITIES:
        return {"error": f"Invalid priority: {priority}. Use: low, medium, high"}
    todos = _load()
    item = {
        "id": uuid4().hex,
        "title": title,
        "priority": priority,
        "status": "pending",
        "created_at": datetime.now().isoformat(),
    }
    todos.append(item)
    _save(todos)
    return {"ok": True, "id": item["id"], "title": title}


@tool(
    name="todo_list",
    description="List todos",
    params={
        "status": {"type": "string", "description": "Filter: pending/done/all", "default": "pending"},
    },
    required=[],
)
async def todo_list(status: str = "pending") -> list[dict]:
    """List todos filtered by status."""
    todos = _load()
    if status != "all":
        todos = [t for t in todos if t["status"] == status]
    return todos


@tool(
    name="todo_done",
    description="Mark a todo as done",
    params={
        "id": {"type": "string", "description": "Todo ID"},
    },
    required=["id"],
)
async def todo_done(id: str) -> dict:
    """Mark a todo as done."""
    todos = _load()
    for t in todos:
        if t["id"] == id:
            t["status"] = "done"
            _save(todos)
            return {"ok": True}
    return {"error": f"Todo {id} not found"}


@tool(
    name="todo_remove",
    description="Remove a todo",
    params={
        "id": {"type": "string", "description": "Todo ID"},
    },
    required=["id"],
)
async def todo_remove(id: str) -> dict:
    """Remove a todo item."""
    todos = _load()
    before = len(todos)
    todos = [t for t in todos if t["id"] != id]
    if len(todos) == before:
        return {"error": f"Todo {id} not found"}
    _save(todos)
    return {"ok": True}
