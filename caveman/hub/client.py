"""Caveman Hub — skill & plugin marketplace client.

Hub API (future):
  GET  /api/skills           — List published skills
  GET  /api/skills/{name}    — Get skill details
  POST /api/skills           — Publish a skill
  GET  /api/plugins          — List plugins
  GET  /api/plugins/{name}   — Get plugin details
  POST /api/plugins          — Publish a plugin
  GET  /api/stats            — Hub statistics

For now: local registry + ClawHub compatibility layer.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Any

import httpx

logger = logging.getLogger(__name__)

# Default hub URL (future)
HUB_URL = "https://hub.cavemanagent.ai"


class HubClient:
    """Client for Caveman Hub — discover, install, and publish skills/plugins."""

    def __init__(self, hub_url: str = HUB_URL, api_key: str = ""):
        self.hub_url = hub_url.rstrip("/")
        self.api_key = api_key
        from caveman.paths import HUB_CACHE_DIR
        self._cache_dir = HUB_CACHE_DIR
        self._cache_dir.mkdir(parents=True, exist_ok=True)

    def _headers(self) -> dict:
        h = {"Content-Type": "application/json"}
        if self.api_key:
            h["Authorization"] = f"Bearer {self.api_key}"
        return h

    async def search_skills(self, query: str = "", limit: int = 20) -> list[dict]:
        """Search for skills on the hub."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{self.hub_url}/api/skills",
                    params={"q": query, "limit": limit},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json().get("skills", [])
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
            logger.warning("Hub unreachable: %s", e)
            return self._search_local_cache(query)

    async def get_skill(self, name: str) -> dict | None:
        """Get skill details from hub."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{self.hub_url}/api/skills/{name}",
                    headers=self._headers(),
                )
                if resp.status_code == 404:
                    return None
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
            return None

    async def install_skill(self, name: str, target_dir: str | None = None) -> bool:
        """Download and install a skill from the hub."""
        skill_data = await self.get_skill(name)
        if not skill_data:
            logger.error("Skill not found: %s", name)
            return False

        from caveman.paths import SKILLS_DIR
        target = Path(target_dir).expanduser() / name if target_dir else SKILLS_DIR / name
        target.mkdir(parents=True, exist_ok=True)

        # Write skill file
        skill_file = target / "skill.yaml"
        import yaml
        with open(skill_file, "w", encoding="utf-8") as f:
            yaml.safe_dump(skill_data, f, default_flow_style=False, allow_unicode=True)

        logger.info("Installed skill: %s → %s", name, target)
        return True

    async def publish_skill(self, skill_path: str) -> dict:
        """Publish a local skill to the hub."""
        path = Path(skill_path).expanduser()
        skill_file = path / "skill.yaml" if path.is_dir() else path

        if not skill_file.exists():
            return {"ok": False, "error": f"Skill file not found: {skill_file}"}

        import yaml
        with open(skill_file, encoding="utf-8") as f:
            skill_data = yaml.safe_load(f)

        skill_data["published_at"] = datetime.now().isoformat()

        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{self.hub_url}/api/skills",
                    headers=self._headers(),
                    json=skill_data,
                )
                resp.raise_for_status()
                return {"ok": True, "result": resp.json()}
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException) as e:
            return {"ok": False, "error": str(e)}

    async def search_plugins(self, query: str = "", limit: int = 20) -> list[dict]:
        """Search for plugins on the hub."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    f"{self.hub_url}/api/plugins",
                    params={"q": query, "limit": limit},
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json().get("plugins", [])
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
            return []

    async def hub_stats(self) -> dict:
        """Get hub statistics."""
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(
                    f"{self.hub_url}/api/stats",
                    headers=self._headers(),
                )
                resp.raise_for_status()
                return resp.json()
        except (httpx.ConnectError, httpx.HTTPStatusError, httpx.TimeoutException):
            return {"status": "offline", "note": "Hub not reachable"}

    def _search_local_cache(self, query: str) -> list[dict]:
        """Fallback: search local cache when hub is offline."""
        cache_file = self._cache_dir / "skills_cache.json"
        if not cache_file.exists():
            return []

        try:
            with open(cache_file, encoding="utf-8") as f:
                skills = json.load(f)
            if query:
                query_lower = query.lower()
                return [s for s in skills if query_lower in s.get("name", "").lower()
                        or query_lower in s.get("description", "").lower()]
            return skills
        except (json.JSONDecodeError, IOError):
            return []


class MigrationTool:
    """Migrate skills/configs between OpenClaw ↔ Caveman ↔ Hermes."""

    @staticmethod
    def from_openclaw_skill(skill_md_path: str) -> dict:
        """Convert an OpenClaw SKILL.md to Caveman skill format."""
        path = Path(skill_md_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Skill file not found: {path}")

        content = path.read_text()
        # Parse SKILL.md structure
        name = path.parent.name
        description = ""
        for line in content.split("\n"):
            if line.startswith("# "):
                name = line[2:].strip()
            elif line.startswith("> ") and not description:
                description = line[2:].strip()
                break

        return {
            "name": name,
            "description": description,
            "content": content,
            "source": "openclaw",
            "original_path": str(path),
            "trigger": "manual",
            "version": 1,
        }

    @staticmethod
    def from_hermes_skill(skill_dict: dict) -> dict:
        """Convert Hermes skill format to Caveman skill format."""
        return {
            "name": skill_dict.get("name", "unnamed"),
            "description": skill_dict.get("description", ""),
            "content": skill_dict.get("content", ""),
            "source": "hermes",
            "trigger": "auto",
            "version": skill_dict.get("version", 1),
        }

    @staticmethod
    def to_openclaw_skill(caveman_skill: dict) -> str:
        """Convert Caveman skill to OpenClaw SKILL.md format."""
        name = caveman_skill.get("name", "skill")
        desc = caveman_skill.get("description", "")
        content = caveman_skill.get("content", "")

        return f"""# {name}

> {desc}

## Steps

{content}

---
*Migrated from Caveman v0.1.0*
"""
