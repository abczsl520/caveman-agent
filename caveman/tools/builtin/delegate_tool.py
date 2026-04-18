"""Delegate tool — spawn a sub-agent to handle a subtask."""
from __future__ import annotations

import logging

from caveman.agent.factory import create_loop
from caveman.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="delegate",
    description="Spawn a sub-agent to handle a subtask",
    params={
        "task": {"type": "string", "description": "Task description for the sub-agent"},
        "max_iterations": {"type": "integer", "description": "Max iterations for sub-agent", "default": 15},
    },
    required=["task"],
)
async def delegate(task: str, max_iterations: int = 15) -> str:
    """Create a sub-agent loop and run the task."""
    try:
        loop = create_loop(max_iterations=max_iterations)
        result = await loop.run(task)
        return result
    except Exception as e:
        logger.error("delegate sub-agent failed: %s", e)
        return f"Sub-agent failed: {e}"
