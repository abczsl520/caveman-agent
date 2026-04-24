"""Cron system — scheduled task execution for Caveman.

Design (inspired by OpenClaw's 59-file cron system, simplified):
  - Jobs stored in SQLite (same DB as sessions)
  - Each job runs in an isolated session
  - Cron expressions parsed via croniter
  - Stagger support to avoid thundering herd
  - Results delivered to configured channel
"""
from __future__ import annotations

import asyncio
import json
import logging
import sqlite3

from caveman.db import connect as db_connect
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Callable, Awaitable

__all__ = ["CronStore", "CronJob", "CronRun", "CronScheduler", "JobStatus"]

logger = logging.getLogger(__name__)

# --- Schema ---

_CRON_SCHEMA = """
CREATE TABLE IF NOT EXISTS cron_jobs (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    schedule TEXT NOT NULL,
    task TEXT NOT NULL,
    channel TEXT DEFAULT NULL,
    channel_id TEXT DEFAULT NULL,
    enabled INTEGER DEFAULT 1,
    max_runtime INTEGER DEFAULT 300,
    created_at TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    last_run_at TEXT DEFAULT NULL,
    last_result TEXT DEFAULT NULL,
    run_count INTEGER DEFAULT 0,
    error_count INTEGER DEFAULT 0,
    extra TEXT DEFAULT '{}'
);

CREATE TABLE IF NOT EXISTS cron_runs (
    id TEXT PRIMARY KEY,
    job_id TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT DEFAULT NULL,
    status TEXT DEFAULT 'running',
    result TEXT DEFAULT NULL,
    tokens_used INTEGER DEFAULT 0,
    duration_s REAL DEFAULT 0,
    FOREIGN KEY (job_id) REFERENCES cron_jobs(id)
);
"""


class JobStatus(Enum):
    """Lifecycle status of a scheduled cron job."""
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"
    SKIPPED = "skipped"


@dataclass
class CronJob:
    """Persistent cron job definition with schedule, command, and delivery target."""
    id: str
    name: str
    schedule: str  # cron expression: "*/5 * * * *"
    task: str      # the prompt to execute
    channel: str | None = None
    channel_id: str | None = None
    enabled: bool = True
    max_runtime: int = 300
    created_at: str = ""
    updated_at: str = ""
    last_run_at: str | None = None
    last_result: str | None = None
    run_count: int = 0
    error_count: int = 0
    extra: dict = field(default_factory=dict)


@dataclass
class CronRun:
    """Record of a single cron job execution with timing and output."""
    id: str
    job_id: str
    started_at: str
    finished_at: str | None = None
    status: str = "running"
    result: str | None = None
    tokens_used: int = 0
    duration_s: float = 0


class CronStore:
    """SQLite-backed cron job storage."""

    def __init__(self, db_path: Path | str):
        self._db_path = str(db_path)
        self._conn = db_connect(self._db_path, row_factory=sqlite3.Row)
        self._conn.executescript(_CRON_SCHEMA)

    def add_job(self, job: CronJob) -> None:
        now = datetime.now(timezone.utc).isoformat()
        if not job.id:
            job.id = str(uuid.uuid4())
        if not job.created_at:
            job.created_at = now
        job.updated_at = now
        self._conn.execute(
            "INSERT INTO cron_jobs (id, name, schedule, task, channel, channel_id, "
            "enabled, max_runtime, created_at, updated_at, extra) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (job.id, job.name, job.schedule, job.task, job.channel, job.channel_id,
             int(job.enabled), job.max_runtime, job.created_at, job.updated_at,
             json.dumps(job.extra)),
        )
        self._conn.commit()

    def get_job(self, job_id: str) -> CronJob | None:
        row = self._conn.execute(
            "SELECT * FROM cron_jobs WHERE id = ?", (job_id,)
        ).fetchone()
        return self._row_to_job(row) if row else None

    def list_jobs(self, enabled_only: bool = False) -> list[CronJob]:
        q = "SELECT * FROM cron_jobs"
        if enabled_only:
            q += " WHERE enabled = 1"
        q += " ORDER BY name"
        return [self._row_to_job(r) for r in self._conn.execute(q).fetchall()]

    def update_job(self, job_id: str, **kwargs) -> bool:
        _ALLOWED = {"name", "schedule", "task", "delivery", "enabled", "extra", "next_run"}
        bad = set(kwargs) - _ALLOWED
        if bad:
            raise ValueError(f"Invalid cron_jobs columns: {bad}")
        sets = []
        vals = []
        for k, v in kwargs.items():
            if k == "extra":
                v = json.dumps(v)
            elif k == "enabled":
                v = int(v)
            sets.append(f"{k} = ?")
            vals.append(v)
        sets.append("updated_at = ?")
        vals.append(datetime.now(timezone.utc).isoformat())
        vals.append(job_id)
        self._conn.execute(
            f"UPDATE cron_jobs SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn.commit()
        return self._conn.total_changes > 0

    def delete_job(self, job_id: str) -> bool:
        self._conn.execute("DELETE FROM cron_runs WHERE job_id = ?", (job_id,))
        self._conn.execute("DELETE FROM cron_jobs WHERE id = ?", (job_id,))
        self._conn.commit()
        return self._conn.total_changes > 0

    def record_run(self, run: CronRun) -> None:
        self._conn.execute(
            "INSERT INTO cron_runs (id, job_id, started_at, finished_at, status, "
            "result, tokens_used, duration_s) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (run.id, run.job_id, run.started_at, run.finished_at,
             run.status, run.result, run.tokens_used, run.duration_s),
        )
        self._conn.commit()

    def update_run(self, run_id: str, **kwargs) -> None:
        _ALLOWED = {"status", "output", "error", "ended_at", "result"}
        bad = set(kwargs) - _ALLOWED
        if bad:
            raise ValueError(f"Invalid cron_runs columns: {bad}")
        sets = [f"{k} = ?" for k in kwargs]
        vals = list(kwargs.values())
        vals.append(run_id)
        self._conn.execute(
            f"UPDATE cron_runs SET {', '.join(sets)} WHERE id = ?", vals
        )
        self._conn.commit()

    def get_recent_runs(self, job_id: str, limit: int = 10) -> list[CronRun]:
        rows = self._conn.execute(
            "SELECT * FROM cron_runs WHERE job_id = ? ORDER BY started_at DESC LIMIT ?",
            (job_id, limit),
        ).fetchall()
        return [CronRun(**dict(r)) for r in rows]

    def _row_to_job(self, row: sqlite3.Row) -> CronJob:
        d = dict(row)
        d["enabled"] = bool(d["enabled"])
        d["extra"] = json.loads(d.get("extra") or "{}")
        return CronJob(**d)

    def close(self) -> None:
        self._conn.close()


# --- Scheduler ---

def _next_run_time(schedule: str, after: datetime | None = None) -> datetime | None:
    """Calculate next run time from cron expression or simple interval."""
    # Try simple interval first (5m, 1h, 30s)
    simple = _simple_interval_next(schedule, after)
    if simple is not None:
        return simple
    try:
        from croniter import croniter
        base = after or datetime.now(timezone.utc)
        cron = croniter(schedule, base)
        return cron.get_next(datetime)
    except ImportError:
        logger.warning("croniter not installed — no cron expression support")
        return None
    except Exception as e:
        logger.error("Invalid cron expression '%s': %s", schedule, e)
        return None


def _simple_interval_next(schedule: str, after: datetime | None = None) -> datetime | None:
    """Fallback: parse simple intervals like '5m', '1h', '30s'."""
    import re
    m = re.match(r'^(\d+)([smh])$', schedule.strip())
    if not m:
        return None
    val, unit = int(m.group(1)), m.group(2)
    secs = val * {'s': 1, 'm': 60, 'h': 3600}[unit]
    base = after or datetime.now(timezone.utc)
    from datetime import timedelta
    return base + timedelta(seconds=secs)


class CronScheduler:
    """Runs cron jobs on schedule."""

    def __init__(
        self,
        store: CronStore,
        executor: Callable[[CronJob], Awaitable[str]],
        stagger_seconds: int = 5,
    ):
        self.store = store
        self.executor = executor
        self.stagger = stagger_seconds
        self._running = False
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        """Start the cron scheduler loop."""
        self._running = True
        self._task = asyncio.create_task(self._loop())
        logger.info("Cron scheduler started")

    async def stop(self) -> None:
        """Stop the cron scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass  # intentional: Exception suppressed
        logger.info("Cron scheduler stopped")

    async def _loop(self):
        """Main scheduler loop — check every 60s for due jobs."""
        while self._running:
            try:
                await self._tick()
            except Exception as e:
                logger.error("Cron scheduler error: %s", e)
            await asyncio.sleep(60)

    async def _tick(self):
        """Check all enabled jobs and run any that are due."""
        now = datetime.now(timezone.utc)
        jobs = self.store.list_jobs(enabled_only=True)
        due = []

        for job in jobs:
            next_time = _next_run_time(
                job.schedule,
                datetime.fromisoformat(job.last_run_at) if job.last_run_at else job.created_at and datetime.fromisoformat(job.created_at),
            )
            if next_time and next_time <= now:
                due.append(job)

        if not due:
            return

        logger.info("Cron: %d job(s) due", len(due))
        for i, job in enumerate(due):
            if i > 0 and self.stagger > 0:
                await asyncio.sleep(self.stagger)
            asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: CronJob):
        """Execute a single cron job with timeout."""
        run_id = str(uuid.uuid4())
        started = datetime.now(timezone.utc).isoformat()
        run = CronRun(id=run_id, job_id=job.id, started_at=started)
        self.store.record_run(run)

        start_time = time.monotonic()
        try:
            result = await asyncio.wait_for(
                self.executor(job),
                timeout=job.max_runtime,
            )
            duration = time.monotonic() - start_time
            self.store.update_run(run_id,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status=JobStatus.SUCCESS.value,
                result=result[:10000] if result else None,
                duration_s=round(duration, 2),
            )
            self.store.update_job(job.id,
                last_run_at=datetime.now(timezone.utc).isoformat(),
                last_result=f"✅ {duration:.1f}s",
                run_count=job.run_count + 1,
            )
            logger.info("Cron job '%s' completed in %.1fs", job.name, duration)

        except asyncio.TimeoutError:
            duration = time.monotonic() - start_time
            self.store.update_run(run_id,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status=JobStatus.TIMEOUT.value,
                result=f"Timeout after {job.max_runtime}s",
                duration_s=round(duration, 2),
            )
            self.store.update_job(job.id,
                last_run_at=datetime.now(timezone.utc).isoformat(),
                last_result=f"⏰ timeout {job.max_runtime}s",
                error_count=job.error_count + 1,
            )
            logger.warning("Cron job '%s' timed out after %ds", job.name, job.max_runtime)

        except Exception as e:
            duration = time.monotonic() - start_time
            self.store.update_run(run_id,
                finished_at=datetime.now(timezone.utc).isoformat(),
                status=JobStatus.FAILED.value,
                result=str(e)[:5000],
                duration_s=round(duration, 2),
            )
            self.store.update_job(job.id,
                last_run_at=datetime.now(timezone.utc).isoformat(),
                last_result=f"❌ {str(e)[:100]}",
                error_count=job.error_count + 1,
            )
            logger.error("Cron job '%s' failed: %s", job.name, e)
