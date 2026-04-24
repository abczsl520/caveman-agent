"""Delegate Tool — sub-agent delegation with concurrency and timeout.

Extracted from Hermes delegate_tool.py (1103 lines).
Key patterns: parallel delegation, result merging, timeout, cancellation.
"""
from __future__ import annotations

import asyncio
import logging
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from caveman.tools.registry import tool

__all__ = [
    "MAX_CONCURRENT",
    "DEFAULT_TIMEOUT",
    "DelegateTask",
    "DelegateManager",
    "get_delegate_manager",
    "set_delegate_agent_fn",
    "delegate_task",
    "delegate_parallel",
]


logger = logging.getLogger("caveman.tools.delegate")

MAX_CONCURRENT = 5
DEFAULT_TIMEOUT = 300  # 5 minutes


@dataclass
class DelegateTask:
    """A single delegation task."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    prompt: str = ""
    model: str = ""
    status: str = "pending"  # pending | running | completed | failed | cancelled
    result: str = ""
    error: str = ""
    started_at: float = 0
    completed_at: float = 0
    duration_ms: float = 0


class DelegateManager:
    """Manages sub-agent delegation with concurrency control."""

    def __init__(
        self,
        agent_fn: Optional[Callable] = None,
        max_concurrent: int = MAX_CONCURRENT,
        default_timeout: float = DEFAULT_TIMEOUT,
    ):
        self._agent_fn = agent_fn
        self._max_concurrent = max_concurrent
        self._default_timeout = default_timeout
        self._semaphore = asyncio.Semaphore(max_concurrent)
        self._active_tasks: Dict[str, DelegateTask] = {}

    async def delegate_single(
        self, prompt: str, model: str = "", timeout: float = 0,
    ) -> DelegateTask:
        """Delegate a single task to a sub-agent."""
        task = DelegateTask(prompt=prompt, model=model)
        self._active_tasks[task.id] = task
        timeout = timeout or self._default_timeout

        async with self._semaphore:
            task.status = "running"
            task.started_at = time.monotonic()
            try:
                if self._agent_fn:
                    result = await asyncio.wait_for(
                        self._agent_fn(prompt, model=model),
                        timeout=timeout,
                    )
                    task.result = str(result) if result else ""
                    task.status = "completed"
                else:
                    task.result = f"[No agent configured] Echo: {prompt}"
                    task.status = "completed"
            except asyncio.TimeoutError:
                task.status = "failed"
                task.error = f"Timeout after {timeout}s"
            except Exception as e:
                task.status = "failed"
                task.error = str(e)
            finally:
                task.completed_at = time.monotonic()
                task.duration_ms = (task.completed_at - task.started_at) * 1000

        return task

    async def delegate_parallel(
        self, tasks: List[Dict[str, str]], timeout: float = 0,
    ) -> List[DelegateTask]:
        """Delegate multiple tasks in parallel."""
        timeout = timeout or self._default_timeout
        coros = [
            self.delegate_single(
                prompt=t.get("prompt", ""),
                model=t.get("model", ""),
                timeout=timeout,
            )
            for t in tasks
        ]
        results = await asyncio.gather(*coros, return_exceptions=True)
        return [
            r if isinstance(r, DelegateTask)
            else DelegateTask(status="failed", error=str(r))
            for r in results
        ]

    def merge_results(self, tasks: List[DelegateTask]) -> str:
        """Merge results from multiple delegate tasks."""
        parts = []
        for i, task in enumerate(tasks, 1):
            if task.status == "completed":
                parts.append(f"[Task {i}] {task.result}")
            else:
                parts.append(f"[Task {i}] FAILED: {task.error}")
        return "\n\n".join(parts)

    def list_active(self) -> List[Dict[str, Any]]:
        return [
            {
                "id": t.id,
                "status": t.status,
                "prompt": t.prompt[:100],
                "duration_ms": round(t.duration_ms, 1),
            }
            for t in self._active_tasks.values()
        ]


# ── Tool Registration ──

_manager: Optional[DelegateManager] = None


def get_delegate_manager() -> DelegateManager:
    global _manager
    if _manager is None:
        _manager = DelegateManager()
    return _manager


def set_delegate_agent_fn(fn: Callable) -> None:
    get_delegate_manager()._agent_fn = fn


@tool(
    name="delegate",
    description="Delegate a task to a sub-agent for parallel or specialized processing",
    params={
        "prompt": {"type": "string", "description": "Task prompt for the sub-agent"},
        "model": {"type": "string", "description": "Model to use (optional)"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 300)"},
    },
    required=["prompt"],
)
async def delegate_task(
    prompt: str, model: str = "", timeout: int = 300,
) -> Dict[str, Any]:
    """Delegate a task to a sub-agent."""
    mgr = get_delegate_manager()
    task = await mgr.delegate_single(prompt, model=model, timeout=timeout)
    return {
        "ok": task.status == "completed",
        "result": task.result,
        "error": task.error,
        "duration_ms": round(task.duration_ms, 1),
    }


@tool(
    name="delegate_parallel",
    description="Delegate multiple tasks in parallel to sub-agents",
    params={
        "tasks": {"type": "array", "items": {"type": "object", "properties": {"prompt": {"type": "string"}, "model": {"type": "string"}}, "required": ["prompt"]}, "description": "List of {prompt, model} objects"},
        "timeout": {"type": "integer", "description": "Timeout per task in seconds"},
    },
    required=["tasks"],
)
async def delegate_parallel(
    tasks: List[Dict[str, str]], timeout: int = 300,
) -> Dict[str, Any]:
    """Delegate multiple tasks in parallel."""
    mgr = get_delegate_manager()
    results = await mgr.delegate_parallel(tasks, timeout=timeout)
    return {
        "ok": all(t.status == "completed" for t in results),
        "results": [
            {"status": t.status, "result": t.result, "error": t.error,
             "duration_ms": round(t.duration_ms, 1)}
            for t in results
        ],
        "merged": mgr.merge_results(results),
    }
