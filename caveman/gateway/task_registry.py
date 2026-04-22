"""Task registry — persistent task tracking with flows and delivery.

Inspired by OpenClaw's task system (6.5K LOC), distilled to essentials.

A Task represents a unit of work (agent run, sub-agent, cron job, etc.)
with lifecycle tracking, delivery state, and flow grouping.

Task lifecycle: created → running → completed/failed/cancelled
Flow: groups related tasks (e.g., a multi-step coding task)
Delivery: tracks whether the user has been notified of task state changes
"""
from __future__ import annotations

import json
import logging
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

__all__ = ["TaskStatus", "TaskRuntime", "DeliveryStatus", "TaskRecord", "TaskFlow", "TaskRegistry"]


logger = logging.getLogger(__name__)


class TaskStatus(str, Enum):
    """Lifecycle status of a background task."""
    CREATED = "created"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    LOST = "lost"  # Process died without reporting


class TaskRuntime(str, Enum):
    """Runtime environment for task execution (local, remote, sandboxed)."""
    AGENT = "agent"
    SUBAGENT = "subagent"
    ACP = "acp"
    CRON = "cron"
    HOOK = "hook"


class DeliveryStatus(str, Enum):
    """Delivery state of a task's output to its target channel."""
    PENDING = "pending"
    DELIVERED = "delivered"
    SUPPRESSED = "suppressed"
    NOT_APPLICABLE = "not_applicable"


@dataclass
class TaskRecord:
    """Persistent record of a background task with status, output, and delivery info."""
    task_id: str
    title: str
    status: TaskStatus = TaskStatus.CREATED
    runtime: TaskRuntime = TaskRuntime.AGENT
    session_id: str = ""
    flow_id: str = ""
    parent_task_id: str = ""

    # Timing
    created_at: float = 0.0
    started_at: float = 0.0
    completed_at: float = 0.0

    # Result
    result_summary: str = ""
    error: str = ""
    progress: float = 0.0  # 0.0 to 1.0

    # Delivery
    delivery_status: DeliveryStatus = DeliveryStatus.PENDING
    delivery_channel: str = ""

    # Metadata
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class TaskFlow:
    """Groups related tasks into a logical flow."""
    flow_id: str
    title: str
    owner_session: str = ""
    created_at: float = 0.0
    task_ids: list[str] = field(default_factory=list)
    status: str = "active"  # active, completed, cancelled
    metadata: dict[str, Any] = field(default_factory=dict)


class TaskRegistry:
    """In-memory task registry with optional SQLite persistence."""

    def __init__(self, persist_dir: Path | None = None) -> None:
        self._tasks: dict[str, TaskRecord] = {}
        self._flows: dict[str, TaskFlow] = {}
        self._persist_dir = persist_dir or (CAVEMAN_HOME / "tasks")
        self._observers: list[Any] = []
        self._retention_seconds = 7 * 24 * 3600  # 7 days

    def create_task(
        self,
        title: str,
        runtime: TaskRuntime = TaskRuntime.AGENT,
        session_id: str = "",
        flow_id: str = "",
        parent_task_id: str = "",
        metadata: dict[str, Any] | None = None,
    ) -> TaskRecord:
        """Create and register a new task."""
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            title=title,
            runtime=runtime,
            session_id=session_id,
            flow_id=flow_id,
            parent_task_id=parent_task_id,
            created_at=time.time(),
            metadata=metadata or {},
        )
        self._tasks[task.task_id] = task

        if flow_id and flow_id in self._flows:
            self._flows[flow_id].task_ids.append(task.task_id)

        self._notify("created", task)
        logger.debug("Task created: %s (%s)", task.task_id[:12], title)
        return task

    def start_task(self, task_id: str) -> bool:
        """Mark a task as running."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.CREATED:
            return False
        task.status = TaskStatus.RUNNING
        task.started_at = time.time()
        self._notify("started", task)
        return True

    def complete_task(self, task_id: str, summary: str = "") -> bool:
        """Mark a task as completed."""
        task = self._tasks.get(task_id)
        if not task or task.status not in (TaskStatus.CREATED, TaskStatus.RUNNING):
            return False
        task.status = TaskStatus.COMPLETED
        task.completed_at = time.time()
        task.result_summary = summary
        task.progress = 1.0
        self._notify("completed", task)
        self._check_flow_completion(task.flow_id)
        return True

    def fail_task(self, task_id: str, error: str = "") -> bool:
        """Mark a task as failed."""
        task = self._tasks.get(task_id)
        if not task or task.status not in (TaskStatus.CREATED, TaskStatus.RUNNING):
            return False
        task.status = TaskStatus.FAILED
        task.completed_at = time.time()
        task.error = error
        self._notify("failed", task)
        return True

    def cancel_task(self, task_id: str) -> bool:
        """Cancel a task."""
        task = self._tasks.get(task_id)
        if not task or _is_terminal(task.status):
            return False
        task.status = TaskStatus.CANCELLED
        task.completed_at = time.time()
        self._notify("cancelled", task)
        return True

    def update_progress(self, task_id: str, progress: float, summary: str = "") -> bool:
        """Update task progress (0.0 to 1.0)."""
        task = self._tasks.get(task_id)
        if not task or task.status != TaskStatus.RUNNING:
            return False
        task.progress = max(0.0, min(1.0, progress))
        if summary:
            task.result_summary = summary
        return True

    def get_task(self, task_id: str) -> TaskRecord | None:
        return self._tasks.get(task_id)

    def list_tasks(
        self,
        session_id: str = "",
        status: TaskStatus | None = None,
        runtime: TaskRuntime | None = None,
        limit: int = 50,
    ) -> list[TaskRecord]:
        """List tasks with optional filters."""
        tasks = list(self._tasks.values())
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        if status:
            tasks = [t for t in tasks if t.status == status]
        if runtime:
            tasks = [t for t in tasks if t.runtime == runtime]
        tasks.sort(key=lambda t: t.created_at, reverse=True)
        return tasks[:limit]

    def summary(self) -> dict[str, Any]:
        """Get registry summary stats."""
        by_status: dict[str, int] = {}
        for t in self._tasks.values():
            by_status[t.status.value] = by_status.get(t.status.value, 0) + 1
        return {
            "total": len(self._tasks),
            "by_status": by_status,
            "flows": len(self._flows),
            "pending_delivery": sum(
                1 for t in self._tasks.values()
                if t.delivery_status == DeliveryStatus.PENDING and _is_terminal(t.status)
            ),
        }

    # ── Flows ──

    def create_flow(self, title: str, owner_session: str = "") -> TaskFlow:
        """Create a task flow."""
        flow = TaskFlow(
            flow_id=str(uuid.uuid4()),
            title=title,
            owner_session=owner_session,
            created_at=time.time(),
        )
        self._flows[flow.flow_id] = flow
        return flow

    def get_flow(self, flow_id: str) -> TaskFlow | None:
        return self._flows.get(flow_id)

    def list_flows(self, status: str = "") -> list[TaskFlow]:
        flows = list(self._flows.values())
        if status:
            flows = [f for f in flows if f.status == status]
        return sorted(flows, key=lambda f: f.created_at, reverse=True)

    # ── Delivery ──

    def mark_delivered(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task:
            return False
        task.delivery_status = DeliveryStatus.DELIVERED
        return True

    def get_pending_deliveries(self, session_id: str = "") -> list[TaskRecord]:
        """Get tasks with pending delivery notifications."""
        tasks = [
            t for t in self._tasks.values()
            if t.delivery_status == DeliveryStatus.PENDING and _is_terminal(t.status)
        ]
        if session_id:
            tasks = [t for t in tasks if t.session_id == session_id]
        return tasks

    # ── Maintenance ──

    def cleanup_old(self) -> int:
        """Remove tasks older than retention period."""
        cutoff = time.time() - self._retention_seconds
        to_remove = [
            tid for tid, t in self._tasks.items()
            if _is_terminal(t.status) and t.completed_at and t.completed_at < cutoff
        ]
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    def mark_lost_tasks(self, timeout: float = 3600) -> int:
        """Mark running tasks as lost if they've been running too long."""
        cutoff = time.time() - timeout
        count = 0
        for task in self._tasks.values():
            if task.status == TaskStatus.RUNNING and task.started_at < cutoff:
                task.status = TaskStatus.LOST
                task.completed_at = time.time()
                task.error = "task timed out (marked as lost)"
                self._notify("lost", task)
                count += 1
        return count

    # ── Observers ──

    def on_event(self, callback: Any) -> None:
        self._observers.append(callback)

    def _notify(self, event: str, task: TaskRecord) -> None:
        for obs in self._observers:
            try:
                obs(event, task)
            except Exception as e:
                logger.debug("Task observer error: %s", e)

    def _check_flow_completion(self, flow_id: str) -> None:
        if not flow_id:
            return
        flow = self._flows.get(flow_id)
        if not flow:
            return
        tasks = [self._tasks.get(tid) for tid in flow.task_ids]
        tasks = [t for t in tasks if t is not None]
        if all(_is_terminal(t.status) for t in tasks):
            flow.status = "completed"

    # ── Persistence ──

    def save(self) -> None:
        """Save registry to disk."""
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        data = {
            "tasks": {
                tid: _task_to_dict(t) for tid, t in self._tasks.items()
            },
            "flows": {
                fid: _flow_to_dict(f) for fid, f in self._flows.items()
            },
        }
        path = self._persist_dir / "registry.json"
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def load(self) -> None:
        """Load registry from disk."""
        path = self._persist_dir / "registry.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for tid, td in data.get("tasks", {}).items():
                self._tasks[tid] = _dict_to_task(td)
            for fid, fd in data.get("flows", {}).items():
                self._flows[fid] = _dict_to_flow(fd)
        except Exception as e:
            logger.warning("Failed to load task registry: %s", e)


def _is_terminal(status: TaskStatus) -> bool:
    return status in (TaskStatus.COMPLETED, TaskStatus.FAILED, TaskStatus.CANCELLED, TaskStatus.LOST)


def _task_to_dict(t: TaskRecord) -> dict[str, Any]:
    return {
        "task_id": t.task_id, "title": t.title, "status": t.status.value,
        "runtime": t.runtime.value, "session_id": t.session_id,
        "flow_id": t.flow_id, "parent_task_id": t.parent_task_id,
        "created_at": t.created_at, "started_at": t.started_at,
        "completed_at": t.completed_at, "result_summary": t.result_summary,
        "error": t.error, "progress": t.progress,
        "delivery_status": t.delivery_status.value,
        "delivery_channel": t.delivery_channel, "metadata": t.metadata,
    }


def _dict_to_task(d: dict[str, Any]) -> TaskRecord:
    return TaskRecord(
        task_id=d["task_id"], title=d["title"],
        status=TaskStatus(d.get("status", "created")),
        runtime=TaskRuntime(d.get("runtime", "agent")),
        session_id=d.get("session_id", ""),
        flow_id=d.get("flow_id", ""),
        parent_task_id=d.get("parent_task_id", ""),
        created_at=d.get("created_at", 0),
        started_at=d.get("started_at", 0),
        completed_at=d.get("completed_at", 0),
        result_summary=d.get("result_summary", ""),
        error=d.get("error", ""),
        progress=d.get("progress", 0),
        delivery_status=DeliveryStatus(d.get("delivery_status", "pending")),
        delivery_channel=d.get("delivery_channel", ""),
        metadata=d.get("metadata", {}),
    )


def _flow_to_dict(f: TaskFlow) -> dict[str, Any]:
    return {
        "flow_id": f.flow_id, "title": f.title,
        "owner_session": f.owner_session, "created_at": f.created_at,
        "task_ids": f.task_ids, "status": f.status, "metadata": f.metadata,
    }


def _dict_to_flow(d: dict[str, Any]) -> TaskFlow:
    return TaskFlow(
        flow_id=d["flow_id"], title=d["title"],
        owner_session=d.get("owner_session", ""),
        created_at=d.get("created_at", 0),
        task_ids=d.get("task_ids", []),
        status=d.get("status", "active"),
        metadata=d.get("metadata", {}),
    )
