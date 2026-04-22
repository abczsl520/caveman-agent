"""Cron integration for the gateway — runs cron jobs as agent tasks."""
from __future__ import annotations

import asyncio
import logging
from typing import Any, Callable

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger("caveman.gateway")


async def start_cron_scheduler(
    config: dict[str, Any],
    get_or_create_session: Callable,
    router: Any,
    store: Any,
) -> None:
    """Start the cron scheduler if any jobs exist.

    Args:
        config: Full config dict.
        get_or_create_session: Async fn(key) -> session dict.
        router: GatewayRouter instance.
        store: SessionDB instance.
    """
    try:
        from caveman.cron import CronStore, CronScheduler
        cron_store = CronStore(CAVEMAN_HOME / "sessions.db")
        jobs = cron_store.list_jobs(enabled_only=True)
        if not jobs:
            logger.debug("No cron jobs configured, scheduler idle")
            cron_store.close()
            return

        async def _execute_job(job):
            """Run a cron job as an isolated agent task."""
            session_key = f"cron:{job.id}"
            session = await get_or_create_session(session_key)
            source = {"gateway": job.channel or "internal",
                      "channel_id": job.channel_id or ""}
            from caveman.gateway.task_runner import run_single_task
            result = await run_single_task(
                task=job.task, session=session,
                gw_name=job.channel or "internal",
                channel_id=job.channel_id or "",
                source_channel=source, router=router,
                store=store, config=config,
            )
            if job.channel and job.channel_id and result:
                try:
                    await router.send(
                        job.channel, job.channel_id,
                        f"⏰ Cron [{job.name}]: {result[:1500]}",
                    )
                except Exception as e:
                    logger.warning("Cron delivery failed for %s: %s", job.name, e)
            return result

        scheduler = CronScheduler(cron_store, _execute_job)
        await scheduler.start()
        logger.info("Cron scheduler started with %d job(s)", len(jobs))
        try:
            while True:
                await asyncio.sleep(3600)
        except asyncio.CancelledError:
            await scheduler.stop()
            cron_store.close()
    except Exception as e:
        logger.error("Cron scheduler failed to start: %s", e)
