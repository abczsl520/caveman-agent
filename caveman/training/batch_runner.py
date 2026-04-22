"""Batch runner — run multiple tasks in parallel for training data generation.

Ported from Hermes batch_runner.py (MIT, Nous Research), simplified for Caveman.
Used by the training pipeline to generate trajectories at scale.
"""
from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from caveman.aio import aio_mkdir

__all__ = ["BatchRunner", "BatchConfig", "BatchResult"]

logger = logging.getLogger(__name__)


@dataclass
class BatchConfig:
    """Configuration for a batch run."""
    dataset_path: str = ""
    output_dir: str = ""
    batch_size: int = 10
    max_concurrent: int = 3
    timeout_per_task: int = 300
    model: str = ""
    resume: bool = False
    run_name: str = ""


@dataclass
class TaskResult:
    """Result of a single task execution."""
    task_id: str
    task: str
    success: bool = False
    result: str = ""
    tool_calls: int = 0
    tokens_used: int = 0
    duration_s: float = 0.0
    error: str = ""


@dataclass
class BatchResult:
    """Aggregate result of a batch run."""
    run_name: str
    total_tasks: int = 0
    completed: int = 0
    failed: int = 0
    total_tokens: int = 0
    total_duration_s: float = 0.0
    results: list[TaskResult] = field(default_factory=list)

    @property
    def success_rate(self) -> float:
        return self.completed / self.total_tasks if self.total_tasks else 0.0


class BatchRunner:
    """Run multiple agent tasks in parallel."""

    def __init__(self, config: BatchConfig):
        self.config = config
        self._checkpoint_path: Path | None = None

    def _load_dataset(self) -> list[dict]:
        """Load tasks from JSONL file."""
        path = Path(self.config.dataset_path)
        if not path.exists():
            raise FileNotFoundError(f"Dataset not found: {path}")
        tasks = []
        for line in path.read_text().splitlines():
            if line.strip():
                tasks.append(json.loads(line))
        return tasks

    def _load_checkpoint(self) -> set[str]:
        """Load completed task IDs from checkpoint."""
        if not self._checkpoint_path or not self._checkpoint_path.exists():
            return set()
        completed = set()
        for line in self._checkpoint_path.read_text().splitlines():
            if line.strip():
                try:
                    data = json.loads(line)
                    completed.add(data.get("task_id", ""))
                except json.JSONDecodeError:
                    pass  # intentional: Exception suppressed
        return completed

    def _save_checkpoint(self, result: TaskResult) -> None:
        """Append a task result to checkpoint file."""
        if self._checkpoint_path:
            with open(self._checkpoint_path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"task_id": result.task_id, "success": result.success}) + "\n")

    async def _run_single_task(self, task_data: dict) -> TaskResult:
        """Run a single task through the agent."""
        from caveman.agent.loop import AgentLoop
        task_id = task_data.get("id", str(hash(task_data.get("task", ""))))
        task_text = task_data.get("task", task_data.get("prompt", ""))
        result = TaskResult(task_id=task_id, task=task_text)
        start = time.monotonic()
        try:
            loop = AgentLoop(model=self.config.model or None)
            response = await asyncio.wait_for(
                loop.run(task_text),
                timeout=self.config.timeout_per_task,
            )
            result.success = True
            result.result = response or ""
            result.tool_calls = getattr(loop, '_tool_call_count', 0)
            try:
                usage = loop.provider.usage_stats
                if isinstance(usage, dict):
                    result.tokens_used = usage.get('total_input_tokens', 0) + usage.get('total_output_tokens', 0)
            except Exception as exc:
                logger.debug("_run_single_task: suppressed %s", exc)
        except asyncio.TimeoutError:
            result.error = f"Timeout after {self.config.timeout_per_task}s"
        except Exception as e:
            result.error = str(e)
        finally:
            result.duration_s = time.monotonic() - start
            await loop.close()
        return result

    async def run(self) -> BatchResult:
        """Run the batch."""
        tasks = self._load_dataset()
        run_name = self.config.run_name or f"batch_{int(time.time())}"
        output_dir = Path(self.config.output_dir or f"batch_runs/{run_name}")
        await aio_mkdir(output_dir, parents=True, exist_ok=True)
        self._checkpoint_path = output_dir / "checkpoint.jsonl"

        # Resume support
        completed_ids = self._load_checkpoint() if self.config.resume else set()
        remaining = [t for t in tasks if t.get("id", str(hash(t.get("task", "")))) not in completed_ids]

        batch_result = BatchResult(run_name=run_name, total_tasks=len(tasks))
        batch_result.completed = len(completed_ids)

        logger.info("Batch run '%s': %d tasks (%d remaining)", run_name, len(tasks), len(remaining))

        # Process in batches with concurrency limit
        sem = asyncio.Semaphore(self.config.max_concurrent)

        async def _limited_run(task_data):
            async with sem:
                return await self._run_single_task(task_data)

        for i in range(0, len(remaining), self.config.batch_size):
            batch = remaining[i:i + self.config.batch_size]
            results = await asyncio.gather(*[_limited_run(t) for t in batch], return_exceptions=True)

            for r in results:
                if isinstance(r, Exception):
                    batch_result.failed += 1
                    continue
                batch_result.results.append(r)
                batch_result.total_tokens += r.tokens_used
                if r.success:
                    batch_result.completed += 1
                else:
                    batch_result.failed += 1
                self._save_checkpoint(r)

            logger.info("Batch progress: %d/%d completed", batch_result.completed, batch_result.total_tasks)

        batch_result.total_duration_s = sum(r.duration_s for r in batch_result.results)

        # Save final results
        results_path = output_dir / "results.jsonl"
        with open(results_path, "w", encoding="utf-8") as f:
            for r in batch_result.results:
                f.write(json.dumps({
                    "task_id": r.task_id, "task": r.task, "success": r.success,
                    "result": r.result[:5000], "tool_calls": r.tool_calls,
                    "tokens": r.tokens_used, "duration_s": round(r.duration_s, 2),
                    "error": r.error,
                }) + "\n")

        logger.info(
            "Batch '%s' complete: %d/%d success (%.1f%%), %d tokens, %.1fs",
            run_name, batch_result.completed, batch_result.total_tasks,
            batch_result.success_rate * 100, batch_result.total_tokens,
            batch_result.total_duration_s,
        )
        return batch_result
