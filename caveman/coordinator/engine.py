"""Coordinator — multi-agent orchestration engine.

The Coordinator manages parallel/sequential agent execution:
  - Task decomposition: Break complex tasks into sub-tasks
  - Agent routing: Assign sub-tasks to best-fit agents
  - Result aggregation: Combine sub-task results
  - Failure handling: Retry, fallback, or escalate

Architecture inspired by:
  - Google ADK hierarchical agent trees
  - OpenClaw sub-agent spawning
  - Mastra composable orchestration

Agents available:
  - caveman (self): General tasks
  - claude-code: Coding tasks (via ACP)
  - hermes: Self-improving tasks (via bridge)
  - web: Browser automation tasks
"""
from __future__ import annotations
import asyncio
import logging
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Awaitable

logger = logging.getLogger(__name__)


class TaskStatus(Enum):
    """Status of a sub-task in the execution plan."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class SubTask:
    """A decomposed sub-task within a coordinated plan."""

    def __init__(
        self,
        task_id: str,
        description: str,
        agent: str = "caveman",
        depends_on: list[str] | None = None,
        timeout: int | None = None,
    ):
        from caveman.paths import DEFAULT_SUBTASK_TIMEOUT
        self.task_id = task_id
        self.description = description
        self.agent = agent
        self.depends_on = depends_on or []
        self.timeout = timeout or DEFAULT_SUBTASK_TIMEOUT
        self.status = TaskStatus.PENDING
        self.result: Any = None
        self.error: str | None = None
        self.started_at: str | None = None
        self.completed_at: str | None = None
        self.verification: str | None = None  # VerifyResult value

    def to_dict(self) -> dict:
        return {
            "task_id": self.task_id,
            "description": self.description,
            "agent": self.agent,
            "depends_on": self.depends_on,
            "status": self.status.value,
            "result": str(self.result)[:200] if self.result else None,
            "error": self.error,
        }


class ExecutionPlan:
    """A plan of sub-tasks to execute."""

    def __init__(self, goal: str):
        self.goal = goal
        self.tasks: dict[str, SubTask] = {}
        self.created_at = datetime.now().isoformat()

    def add_task(self, task: SubTask) -> None:
        self.tasks[task.task_id] = task

    def get_ready_tasks(self) -> list[SubTask]:
        """Get tasks whose dependencies are all completed."""
        ready = []
        for task in self.tasks.values():
            if task.status != TaskStatus.PENDING:
                continue
            deps_met = all(
                dep in self.tasks and self.tasks[dep].status == TaskStatus.COMPLETED
                for dep in task.depends_on
            )
            if deps_met:
                ready.append(task)
        return ready

    def has_deadlock(self) -> bool:
        """Detect if pending tasks can never become ready.

        True when there are pending tasks but none are ready — caused by
        missing deps, circular deps, or deps stuck in FAILED/CANCELLED.
        """
        pending = [
            t for t in self.tasks.values() if t.status == TaskStatus.PENDING
        ]
        if not pending:
            return False
        return len(self.get_ready_tasks()) == 0

    @property
    def is_complete(self) -> bool:
        return all(
            t.status in (TaskStatus.COMPLETED, TaskStatus.CANCELLED)
            for t in self.tasks.values()
        )

    @property
    def has_failures(self) -> bool:
        return any(t.status == TaskStatus.FAILED for t in self.tasks.values())

    def summary(self) -> dict:
        by_status = {}
        for task in self.tasks.values():
            by_status[task.status.value] = by_status.get(task.status.value, 0) + 1
        return {
            "goal": self.goal,
            "total_tasks": len(self.tasks),
            "by_status": by_status,
            "is_complete": self.is_complete,
        }


class Coordinator:
    """Multi-agent task coordinator.

    Usage:
        coord = Coordinator()
        coord.register_agent("claude-code", my_claude_code_fn)
        coord.register_agent("hermes", my_hermes_fn)

        plan = coord.plan("Build and deploy a web app")
        results = await coord.execute(plan)
    """

    def __init__(
        self,
        max_parallel: int = 3,
        verifier=None,  # VerificationAgent instance
        verify_tasks: bool = True,
    ):
        self.max_parallel = max_parallel
        self._agents: dict[str, Callable[[str, dict], Awaitable[Any]]] = {}
        self._plans: list[ExecutionPlan] = []
        self._verifier = verifier
        self._verify_tasks = verify_tasks

    def register_agent(
        self, name: str, handler: Callable[[str, dict], Awaitable[Any]]
    ) -> None:
        """Register an agent handler.

        Handler signature: async def handler(task: str, context: dict) -> Any
        """
        self._agents[name] = handler
        logger.info("Coordinator: registered agent '%s'", name)

    def plan(self, goal: str, tasks: list[dict] | None = None) -> ExecutionPlan:
        """Create an execution plan.

        If tasks not provided, creates a single-task plan.
        Task dict format: {"id": str, "description": str, "agent": str, "depends_on": [str]}
        """
        ep = ExecutionPlan(goal)

        if tasks:
            for t in tasks:
                ep.add_task(SubTask(
                    task_id=t["id"],
                    description=t["description"],
                    agent=t.get("agent", "caveman"),
                    depends_on=t.get("depends_on", []),
                    timeout=t.get("timeout"),  # None = SubTask uses DEFAULT_SUBTASK_TIMEOUT
                ))
        else:
            ep.add_task(SubTask(task_id="main", description=goal))

        self._plans.append(ep)
        return ep

    async def execute(
        self, plan: ExecutionPlan, context: dict | None = None
    ) -> dict[str, Any]:
        """Execute a plan, respecting dependencies and parallelism."""
        ctx = context or {}
        semaphore = asyncio.Semaphore(self.max_parallel)

        while not plan.is_complete:
            ready = plan.get_ready_tasks()
            if not ready:
                if plan.has_failures or plan.has_deadlock():
                    logger.error("Plan stalled (failures or deadlock) — aborting")
                    break
                # Shouldn't happen if plan is well-formed
                await asyncio.sleep(0.1)
                continue

            # Execute ready tasks in parallel (up to max_parallel)
            coros = [self._run_task(task, ctx, semaphore) for task in ready]
            await asyncio.gather(*coros, return_exceptions=True)

        # Collect results
        results = {}
        for task in plan.tasks.values():
            results[task.task_id] = {
                "status": task.status.value,
                "result": task.result,
                "error": task.error,
            }

        return {
            "goal": plan.goal,
            "tasks": results,
            "summary": plan.summary(),
        }

    async def _run_task(
        self, task: SubTask, context: dict, semaphore: asyncio.Semaphore
    ) -> None:
        """Run a single sub-task."""
        async with semaphore:
            task.status = TaskStatus.RUNNING
            task.started_at = datetime.now().isoformat()

            handler = self._agents.get(task.agent)
            if not handler:
                task.status = TaskStatus.FAILED
                task.error = f"No handler for agent: {task.agent}"
                return

            try:
                task.result = await asyncio.wait_for(
                    handler(task.description, context),
                    timeout=task.timeout,
                )
                # Verification step (anti-rationalization)
                if self._verify_tasks and self._verifier:
                    vr = await self._verifier.verify(
                        task.description, task.result, context,
                    )
                    task.verification = vr.result.value
                    if not vr.passed:
                        task.status = TaskStatus.FAILED
                        task.error = f"Verification failed: {vr.summary}"
                        logger.warning(
                            "Task '%s' failed verification: %s",
                            task.task_id, vr.summary,
                        )
                        return

                task.status = TaskStatus.COMPLETED
                task.completed_at = datetime.now().isoformat()
                logger.info("Task '%s' completed", task.task_id)
            except asyncio.TimeoutError:
                task.status = TaskStatus.FAILED
                task.error = f"Timeout after {task.timeout}s"
                logger.warning("Task '%s' timed out", task.task_id)
            except Exception as e:
                task.status = TaskStatus.FAILED
                task.error = str(e)
                logger.error("Task '%s' failed: %s", task.task_id, e)
