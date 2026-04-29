"""ACP Client — connect to another ACP-compatible agent."""
from __future__ import annotations

import asyncio
import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


def _json_object(resp: httpx.Response) -> dict[str, Any]:
    data = resp.json()
    if isinstance(data, dict):
        return data
    return {"error": "ACP response must be a JSON object"}


class ACPClient:
    """Connect to a remote ACP-compatible agent and send tasks."""

    def __init__(self, base_url: str, timeout: float = 30.0):
        self.base_url = base_url.rstrip("/")
        self._client = httpx.AsyncClient(timeout=timeout)

    async def send_task(
        self, message: str, metadata: dict | None = None, poll_interval: float = 1.0,
    ) -> dict:
        """Send a task and poll until completion."""
        resp = await self._client.post(
            f"{self.base_url}/acp/v1/tasks",
            json={
                "message": {"role": "user", "parts": [{"type": "text", "text": message}]},
                "metadata": metadata or {},
            },
        )
        resp.raise_for_status()
        task = _json_object(resp)

        while task.get("status") in ("pending", "running"):
            task_id = task.get("id")
            if not isinstance(task_id, str):
                return {"error": "ACP task response missing string id"}
            await asyncio.sleep(poll_interval)
            resp = await self._client.get(f"{self.base_url}/acp/v1/tasks/{task_id}")
            resp.raise_for_status()
            task = _json_object(resp)

        return task

    async def send_task_async(self, message: str, metadata: dict | None = None) -> str:
        """Send a task without waiting. Returns task ID."""
        resp = await self._client.post(
            f"{self.base_url}/acp/v1/tasks",
            json={
                "message": {"role": "user", "parts": [{"type": "text", "text": message}]},
                "metadata": metadata or {},
            },
        )
        resp.raise_for_status()
        task = _json_object(resp)
        task_id = task.get("id")
        if isinstance(task_id, str):
            return task_id
        raise ValueError("ACP task response missing string id")

    async def get_task(self, task_id: str) -> dict:
        """Get task status/result."""
        resp = await self._client.get(f"{self.base_url}/acp/v1/tasks/{task_id}")
        resp.raise_for_status()
        return _json_object(resp)

    async def cancel_task(self, task_id: str) -> dict:
        """Cancel a running task."""
        resp = await self._client.post(f"{self.base_url}/acp/v1/tasks/{task_id}/cancel")
        resp.raise_for_status()
        return _json_object(resp)

    async def close(self) -> None:
        await self._client.aclose()
