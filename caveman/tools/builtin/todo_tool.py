"""Todo tool — persistent task list stored as JSON."""
from __future__ import annotations

import json
import logging
from typing import Any
from datetime import datetime
from uuid import uuid4

from caveman.paths import CAVEMAN_HOME
from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

__all__ = [
    "todo_add",
    "todo_list",
    "todo_finish",
    "todo_remove",
]


_TODO_FILE = CAVEMAN_HOME / "todos.json"


class _TodoFileError(ValueError):
    """Raised when todos.json exists but has an invalid shape."""


def _load() -> list[dict[str, Any]]:
    if _TODO_FILE.exists():
        try:
            data = json.loads(_TODO_FILE.read_text())
        except (json.JSONDecodeError, OSError):
            logger.warning("Corrupt todos.json, starting fresh")
        else:
            if not isinstance(data, list):
                logger.warning("Invalid todos.json shape, starting fresh")
                return []
            if not all(isinstance(item, dict) for item in data):
                raise _TodoFileError("todos.json must contain a list of objects")
            return data
    return []


def _load_or_error() -> tuple[list[dict[str, Any]] | None, dict[str, str] | None]:
    try:
        return _load(), None
    except _TodoFileError as exc:
        return None, {"error": str(exc)}


def _save(todos: list[dict[str, Any]]) -> None:
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
    todos, error = _load_or_error()
    if error:
        return error
    assert todos is not None
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
        "status": {"type": "string", "description": "Filter: pending/finished/all", "default": "pending"},
    },
    required=[],
)
async def todo_list(status: str = "pending") -> list[dict[str, Any]]:
    """List todos filtered by status."""
    todos, error = _load_or_error()
    if error:
        return [error]
    assert todos is not None
    if status != "all":
        todos = [t for t in todos if t.get("status") == status]
    return todos


@tool(
    name="todo_finish",
    description="Mark a todo item as finished",
    params={
        "id": {"type": "string", "description": "Todo ID"},
    },
    required=["id"],
)
async def todo_finish(id: str) -> dict:
    """Mark a todo item as finished."""
    todos, error = _load_or_error()
    if error:
        return error
    assert todos is not None
    for t in todos:
        if t.get("id") == id:
            t["status"] = "finished"
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
    todos, error = _load_or_error()
    if error:
        return error
    assert todos is not None
    before = len(todos)
    todos = [t for t in todos if t.get("id") != id]
    if len(todos) == before:
        return {"error": f"Todo {id} not found"}
    _save(todos)
    return {"ok": True}


# Backward-compatible Python alias; registry exposes only todo_finish to the LLM.
todo_done = todo_finish
