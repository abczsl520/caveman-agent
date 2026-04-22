"""Cronjob Tools — scheduled task management.

Provides cron-like scheduling for recurring agent tasks.
Extracted from Hermes tools/cronjob_tools.py (542 lines).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

__all__ = [
    "CronJob",
    "parse_interval",
    "CronManager",
]


logger = logging.getLogger("caveman.tools.cronjob")

_CRON_DIR = Path.home() / ".caveman" / "cron"


@dataclass
class CronJob:
    """A scheduled job."""
    id: str
    name: str
    schedule: str  # cron expression or interval
    command: str = ""
    task: str = ""
    enabled: bool = True
    created_at: float = 0
    last_run: float = 0
    next_run: float = 0
    run_count: int = 0
    last_result: str = ""
    last_error: str = ""
    max_retries: int = 3
    timeout: int = 300

    @property
    def is_overdue(self) -> bool:
        return self.enabled and self.next_run > 0 and time.time() > self.next_run


def parse_interval(schedule: str) -> Optional[float]:
    """Parse an interval string like '5m', '1h', '30s', '1d'."""
    import re
    match = re.match(r"^(\d+)(s|m|h|d)$", schedule.strip())
    if not match:
        return None
    value = int(match.group(1))
    unit = match.group(2)
    multipliers = {"s": 1, "m": 60, "h": 3600, "d": 86400}
    return value * multipliers[unit]


class CronManager:
    """Manages scheduled jobs."""

    def __init__(self, persist_dir: Optional[Path] = None):
        self._jobs: Dict[str, CronJob] = {}
        self._persist_dir = persist_dir or _CRON_DIR
        self._load()

    def add(
        self,
        name: str,
        schedule: str,
        command: str = "",
        task: str = "",
        **kwargs,
    ) -> CronJob:
        """Add a new cron job."""
        import hashlib
        job_id = hashlib.sha256(f"{name}:{time.time()}".encode()).hexdigest()[:12]

        # Calculate next run
        interval = parse_interval(schedule)
        next_run = time.time() + interval if interval else 0

        job = CronJob(
            id=job_id,
            name=name,
            schedule=schedule,
            command=command,
            task=task,
            created_at=time.time(),
            next_run=next_run,
            **kwargs,
        )
        self._jobs[job_id] = job
        self._save()
        return job

    def remove(self, job_id: str) -> bool:
        if job_id in self._jobs:
            del self._jobs[job_id]
            self._save()
            return True
        return False

    def get(self, job_id: str) -> Optional[CronJob]:
        return self._jobs.get(job_id)

    def list_jobs(self) -> List[CronJob]:
        return list(self._jobs.values())

    def get_overdue(self) -> List[CronJob]:
        """Get jobs that are overdue for execution."""
        return [j for j in self._jobs.values() if j.is_overdue]

    def mark_run(self, job_id: str, result: str = "", error: str = "") -> None:
        """Mark a job as having been run."""
        job = self._jobs.get(job_id)
        if not job:
            return

        job.last_run = time.time()
        job.run_count += 1
        job.last_result = result[:1000]
        job.last_error = error[:500]

        # Calculate next run
        interval = parse_interval(job.schedule)
        if interval:
            job.next_run = time.time() + interval

        self._save()

    def enable(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.enabled = True
            self._save()
            return True
        return False

    def disable(self, job_id: str) -> bool:
        job = self._jobs.get(job_id)
        if job:
            job.enabled = False
            self._save()
            return True
        return False

    def _save(self) -> None:
        self._persist_dir.mkdir(parents=True, exist_ok=True)
        path = self._persist_dir / "jobs.json"
        try:
            data = {}
            for jid, job in self._jobs.items():
                data[jid] = {
                    "id": job.id, "name": job.name, "schedule": job.schedule,
                    "command": job.command, "task": job.task, "enabled": job.enabled,
                    "created_at": job.created_at, "last_run": job.last_run,
                    "next_run": job.next_run, "run_count": job.run_count,
                    "last_result": job.last_result, "last_error": job.last_error,
                    "max_retries": job.max_retries, "timeout": job.timeout,
                }
            path.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
        except Exception as e:
            logger.debug("Failed to save cron jobs: %s", e)

    def _load(self) -> None:
        path = self._persist_dir / "jobs.json"
        if not path.exists():
            return
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            for jid, d in data.items():
                self._jobs[jid] = CronJob(**d)
        except Exception as e:
            logger.debug("Failed to load cron jobs: %s", e)
