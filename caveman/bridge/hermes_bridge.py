"""Hermes bridge — interop with Hermes Agent for skill/trajectory exchange.

Hermes Agent API (v0.7+):
- POST /api/delegate: Delegate task to Hermes
- GET  /api/skills: List skills
- GET  /api/skills/{name}: Get skill details
- POST /api/skills: Create/update skill
- POST /api/trajectories: Export trajectory
- POST /api/trajectories/import: Import trajectory for training
- GET  /api/health: Health check
"""
from __future__ import annotations

import logging
from typing import Any

import httpx

logger = logging.getLogger(__name__)


class HermesBridge:
    """Bridge to Hermes Agent. Enables bidirectional skill/trajectory exchange."""

    def __init__(self, base_url: str = "", api_key: str = ""):
        if not base_url:
            from caveman.paths import DEFAULT_HERMES_URL
            base_url = DEFAULT_HERMES_URL
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self._client: httpx.AsyncClient | None = None
        self._connected = False

    def _headers(self) -> dict[str, str]:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def connect(self) -> bool:
        """Connect and verify Hermes is available."""
        # Close existing client if any (audit fix: prevent leak on reconnect)
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug("Suppressed in hermes_bridge: %s", e)

        self._client = httpx.AsyncClient(timeout=60.0)
        try:
            resp = await self._client.get(
                f"{self.base_url}/api/health", headers=self._headers()
            )
            self._connected = resp.status_code == 200
            return self._connected
        except (httpx.ConnectError, httpx.TimeoutException, httpx.NetworkError) as e:
            # Audit fix: catch all common network errors, not just ConnectError
            logger.warning("Hermes connect failed: %s", e)
            return False
        except Exception as e:
            logger.error("Hermes connect unexpected error: %s", e)
            return False

    async def disconnect(self) -> None:
        if self._client:
            try:
                await self._client.aclose()
            except Exception as e:
                logger.debug("Suppressed in disconnect: %s", e)
            self._client = None
        self._connected = False

    # --- Task Delegation ---

    async def delegate(self, task: str, context: dict[str, Any] | None = None) -> str:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.post(
            f"{self.base_url}/api/delegate",
            headers=self._headers(),
            json={"task": task, "context": context or {}},
        )
        resp.raise_for_status()
        return resp.json().get("result", "")

    # --- Skill Exchange ---

    async def list_skills(self) -> list[dict]:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.get(
            f"{self.base_url}/api/skills", headers=self._headers()
        )
        resp.raise_for_status()
        return resp.json().get("skills", [])

    async def import_skill(self, skill_name: str) -> dict:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.get(
            f"{self.base_url}/api/skills/{skill_name}", headers=self._headers()
        )
        resp.raise_for_status()
        hermes_skill = resp.json()
        return {
            "name": hermes_skill.get("name", skill_name),
            "description": hermes_skill.get("description", ""),
            "content": hermes_skill.get("content", ""),
            "trigger": "auto",
            "source": "hermes",
            "version": hermes_skill.get("version", 1),
            "quality_gates": [
                {"name": "imported_check", "check": "no_errors", "severity": "warn"},
            ],
        }

    async def export_skill(self, skill_dict: dict) -> bool:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.post(
            f"{self.base_url}/api/skills",
            headers=self._headers(),
            json={
                "name": skill_dict.get("name"),
                "description": skill_dict.get("description"),
                "content": skill_dict.get("content", ""),
                "version": skill_dict.get("version", 1),
            },
        )
        return resp.status_code in (200, 201)

    # --- Trajectory Exchange ---

    async def export_trajectory(self, trajectory: list[dict], metadata: dict | None = None) -> bool:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.post(
            f"{self.base_url}/api/trajectories",
            headers=self._headers(),
            json={"trajectory": trajectory, "metadata": metadata or {}},
        )
        return resp.status_code == 200

    async def import_trajectories(self, limit: int = 100) -> list[dict]:
        self._ensure_connected()
        assert self._client is not None
        resp = await self._client.get(
            f"{self.base_url}/api/trajectories",
            headers=self._headers(),
            params={"limit": limit},
        )
        resp.raise_for_status()
        return resp.json().get("trajectories", [])

    def _ensure_connected(self):
        if not self._connected or not self._client:
            raise RuntimeError("Not connected to Hermes. Call connect() first.")
