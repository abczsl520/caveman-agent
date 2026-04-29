"""Browser Providers — cloud browser backends (BrowserBase, Browser-Use).

Provides pluggable cloud browser backends for headless browsing.
Extracted from Hermes browser_providers/ (608 lines).
"""
from __future__ import annotations

import json
import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = [
    "CloudBrowserProvider",
    "BrowserSession",
    "BrowserbaseProvider",
    "BrowserUseProvider",
    "get_provider",
    "list_providers",
]


logger = logging.getLogger("caveman.tools.browser_providers")


class CloudBrowserProvider(ABC):
    """Base class for cloud browser providers."""

    @abstractmethod
    def provider_name(self) -> str: ...

    @abstractmethod
    def is_configured(self) -> bool: ...

    @abstractmethod
    def create_session(self, **kwargs) -> Dict[str, Any]: ...

    @abstractmethod
    def close_session(self, session_id: str) -> bool: ...

    def get_connect_url(self, session_id: str) -> str:
        return ""


@dataclass
class BrowserSession:
    """A cloud browser session."""
    session_id: str
    provider: str
    connect_url: str = ""
    created_at: float = 0
    status: str = "active"
    metadata: Dict[str, Any] = field(default_factory=dict)


class BrowserbaseProvider(CloudBrowserProvider):
    """Browserbase (https://browserbase.com) cloud browser backend."""

    def provider_name(self) -> str:
        return "Browserbase"

    def is_configured(self) -> bool:
        return bool(
            os.environ.get("BROWSERBASE_API_KEY")
            and os.environ.get("BROWSERBASE_PROJECT_ID")
        )

    def create_session(self, **kwargs) -> Dict[str, Any]:
        import urllib.request
        api_key = os.environ.get("BROWSERBASE_API_KEY", "")
        project_id = os.environ.get("BROWSERBASE_PROJECT_ID", "")
        base_url = os.environ.get("BROWSERBASE_BASE_URL", "https://api.browserbase.com").rstrip("/")

        payload = json.dumps({
            "projectId": project_id,
            **kwargs,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "X-BB-API-Key": api_key,
        }
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/sessions",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            session_id = data.get("id", "")
            return {
                "success": True,
                "session_id": session_id,
                "connect_url": f"wss://connect.browserbase.com?apiKey={api_key}&sessionId={session_id}",
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_session(self, session_id: str) -> bool:
        import urllib.request
        api_key = os.environ.get("BROWSERBASE_API_KEY", "")
        base_url = os.environ.get("BROWSERBASE_BASE_URL", "https://api.browserbase.com").rstrip("/")
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/sessions/{session_id}",
                headers={"X-BB-API-Key": api_key},
                method="DELETE",
            )
            with urllib.request.urlopen(req, timeout=10) as resp:
                return bool(resp.status < 300)
        except Exception as e:
            logger.debug("suppressed: %s", e)
            return False

    def get_connect_url(self, session_id: str) -> str:
        api_key = os.environ.get("BROWSERBASE_API_KEY", "")
        return f"wss://connect.browserbase.com?apiKey={api_key}&sessionId={session_id}"


class BrowserUseProvider(CloudBrowserProvider):
    """Browser-Use (https://browser-use.com) AI browser agent backend."""

    def provider_name(self) -> str:
        return "Browser-Use"

    def is_configured(self) -> bool:
        return bool(os.environ.get("BROWSER_USE_API_KEY"))

    def create_session(self, task: str = "", **kwargs) -> Dict[str, Any]:
        import urllib.request
        api_key = os.environ.get("BROWSER_USE_API_KEY", "")
        base_url = os.environ.get("BROWSER_USE_BASE_URL", "https://api.browser-use.com").rstrip("/")

        payload = json.dumps({
            "task": task,
            **kwargs,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {api_key}",
        }
        try:
            req = urllib.request.Request(
                f"{base_url}/v1/tasks",
                data=payload, headers=headers, method="POST",
            )
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = json.loads(resp.read())
            return {
                "success": True,
                "task_id": data.get("id", ""),
                "status": data.get("status", "pending"),
            }
        except Exception as e:
            return {"success": False, "error": str(e)}

    def close_session(self, session_id: str) -> bool:
        # Browser-Use tasks auto-complete
        return True


# ── Provider Registry ──

_PROVIDERS: Dict[str, CloudBrowserProvider] = {
    "browserbase": BrowserbaseProvider(),
    "browser-use": BrowserUseProvider(),
}


def get_provider(name: str = "") -> Optional[CloudBrowserProvider]:
    """Get a browser provider by name, or first available."""
    if name:
        return _PROVIDERS.get(name)
    for provider in _PROVIDERS.values():
        if provider.is_configured():
            return provider
    return None


def list_providers() -> List[Dict[str, Any]]:
    """List all providers with their status."""
    return [
        {
            "name": p.provider_name(),
            "configured": p.is_configured(),
        }
        for p in _PROVIDERS.values()
    ]
