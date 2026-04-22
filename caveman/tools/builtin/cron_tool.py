"""Cron tool — manage scheduled tasks."""
from __future__ import annotations

import logging
from caveman.tools.registry import tool

logger = logging.getLogger(__name__)


@tool(
    name="cron",
    description=(
        "Manage scheduled tasks (cron jobs). "
        "Actions: list, create, get, update, delete, runs. "
        "Schedule: cron expression ('*/5 * * * *') or simple interval ('5m', '1h')."
    ),
    params={
        "action": {
            "type": "string",
            "enum": ["list", "create", "get", "update", "delete", "runs"],
            "description": "Action to perform",
        },
        "job_id": {"type": "string", "description": "Job ID (for get/update/delete/runs)"},
        "name": {"type": "string", "description": "Job name (for create)"},
        "schedule": {"type": "string", "description": "Cron expression or interval"},
        "task": {"type": "string", "description": "Task prompt to execute"},
        "channel": {"type": "string", "description": "Delivery channel: discord/telegram"},
        "channel_id": {"type": "string", "description": "Channel ID for delivery"},
        "enabled": {"type": "boolean", "description": "Enable/disable job"},
        "max_runtime": {"type": "integer", "description": "Max runtime in seconds"},
    },
    required=["action"],
)
async def cron_tool(args: dict, source: dict | None = None) -> dict:
    from caveman.cron import CronStore, CronJob
    from caveman.paths import CAVEMAN_HOME

    store = CronStore(CAVEMAN_HOME / "sessions.db")
    action = args.get("action", "list")

    try:
        if action == "list":
            jobs = store.list_jobs()
            if not jobs:
                return {"ok": True, "jobs": [], "message": "No cron jobs."}
            return {"ok": True, "jobs": [
                {"id": j.id, "name": j.name, "schedule": j.schedule,
                 "enabled": j.enabled, "last_run": j.last_run_at,
                 "last_result": j.last_result, "runs": j.run_count}
                for j in jobs
            ]}

        elif action == "create":
            name, schedule, task = args.get("name"), args.get("schedule"), args.get("task")
            if not all([name, schedule, task]):
                return {"ok": False, "error": "name, schedule, and task required"}
            job = CronJob(
                id="", name=name, schedule=schedule, task=task,
                channel=args.get("channel"),
                channel_id=args.get("channel_id") or (source or {}).get("channel_id"),
                max_runtime=args.get("max_runtime", 300),
            )
            store.add_job(job)
            return {"ok": True, "job_id": job.id, "message": f"Created '{name}'"}

        elif action == "get":
            job = store.get_job(args.get("job_id", ""))
            if not job:
                return {"ok": False, "error": "Job not found"}
            return {"ok": True, "job": {
                "id": job.id, "name": job.name, "schedule": job.schedule,
                "task": job.task, "channel": job.channel, "enabled": job.enabled,
                "max_runtime": job.max_runtime, "last_run": job.last_run_at,
                "runs": job.run_count, "errors": job.error_count,
            }}

        elif action == "update":
            job_id = args.get("job_id", "")
            updates = {k: args[k] for k in
                       ("name", "schedule", "task", "channel", "channel_id", "enabled", "max_runtime")
                       if k in args and args[k] is not None}
            if not updates:
                return {"ok": False, "error": "No updates"}
            store.update_job(job_id, **updates)
            return {"ok": True, "message": f"Updated '{job_id}'"}

        elif action == "delete":
            if store.delete_job(args.get("job_id", "")):
                return {"ok": True, "message": "Deleted"}
            return {"ok": False, "error": "Not found"}

        elif action == "runs":
            runs = store.get_recent_runs(args.get("job_id", ""))
            return {"ok": True, "runs": [
                {"id": r.id, "started": r.started_at, "status": r.status,
                 "duration": r.duration_s, "result": (r.result or "")[:200]}
                for r in runs
            ]}

        return {"ok": False, "error": f"Unknown action: {action}"}
    finally:
        store.close()
