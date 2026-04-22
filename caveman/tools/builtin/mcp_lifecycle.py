"""MCP Server Lifecycle — health checks, auto-reconnect, env safety.

Extracted from Hermes mcp_tool.py MCPServerTask (480 lines).
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Set

__all__ = [
    "BLOCKED_ENV_VARS",
    "SECRET_PATTERNS",
    "build_safe_env",
    "ServerHealth",
    "MCPServerConfig",
    "MCPServerLifecycle",
]


logger = logging.getLogger("caveman.tools.mcp_lifecycle")

# Environment variables that should never be passed to MCP servers
BLOCKED_ENV_VARS = frozenset({
    "AWS_SECRET_ACCESS_KEY", "AWS_SESSION_TOKEN",
    "OPENAI_API_KEY", "ANTHROPIC_API_KEY",
    "GITHUB_TOKEN", "GH_TOKEN",
    "DATABASE_URL", "DB_PASSWORD",
    "STRIPE_SECRET_KEY", "STRIPE_API_KEY",
    "SENDGRID_API_KEY", "TWILIO_AUTH_TOKEN",
    "SLACK_TOKEN", "SLACK_BOT_TOKEN",
    "DISCORD_TOKEN", "TELEGRAM_BOT_TOKEN",
    "SSH_PRIVATE_KEY", "GPG_PASSPHRASE",
})

# Patterns for env vars that look like secrets
SECRET_PATTERNS = [
    re.compile(r".*_SECRET.*", re.IGNORECASE),
    re.compile(r".*_TOKEN$", re.IGNORECASE),
    re.compile(r".*_KEY$", re.IGNORECASE),
    re.compile(r".*_PASSWORD$", re.IGNORECASE),
    re.compile(r".*_CREDENTIAL.*", re.IGNORECASE),
]


def build_safe_env(user_env: Optional[Dict[str, str]] = None) -> Dict[str, str]:
    """Build a safe environment for MCP server processes.

    Filters out known secret env vars and anything matching secret patterns.
    User-provided env vars are always included (they're explicit).
    """
    safe = {}
    # Allowlist of safe system env vars
    SAFE_VARS = {
        "PATH", "HOME", "USER", "SHELL", "LANG", "LC_ALL", "LC_CTYPE",
        "TERM", "TMPDIR", "TMP", "TEMP", "XDG_RUNTIME_DIR",
        "NODE_PATH", "PYTHONPATH", "GOPATH", "CARGO_HOME", "RUSTUP_HOME",
    }

    for key, value in os.environ.items():
        if key in SAFE_VARS:
            safe[key] = value
        elif key in BLOCKED_ENV_VARS:
            continue
        elif any(p.match(key) for p in SECRET_PATTERNS):
            continue
        else:
            safe[key] = value

    # User-provided env always wins
    if user_env:
        safe.update(user_env)

    return safe


@dataclass
class ServerHealth:
    """Health status of an MCP server."""
    connected: bool = False
    last_ping_at: float = 0
    last_pong_at: float = 0
    consecutive_failures: int = 0
    total_requests: int = 0
    total_errors: int = 0
    uptime_start: float = 0

    @property
    def is_healthy(self) -> bool:
        if not self.connected:
            return False
        if self.consecutive_failures >= 3:
            return False
        return True

    @property
    def uptime_seconds(self) -> float:
        if not self.connected or not self.uptime_start:
            return 0
        return time.monotonic() - self.uptime_start

    def record_success(self) -> None:
        self.consecutive_failures = 0
        self.total_requests += 1

    def record_failure(self) -> None:
        self.consecutive_failures += 1
        self.total_errors += 1
        self.total_requests += 1

    def to_dict(self) -> Dict[str, Any]:
        return {
            "connected": self.connected,
            "healthy": self.is_healthy,
            "consecutive_failures": self.consecutive_failures,
            "total_requests": self.total_requests,
            "total_errors": self.total_errors,
            "uptime_seconds": round(self.uptime_seconds, 1),
            "error_rate": round(self.total_errors / max(self.total_requests, 1), 3),
        }


@dataclass
class MCPServerConfig:
    """Configuration for an MCP server."""
    name: str
    command: str = ""
    args: List[str] = field(default_factory=list)
    url: str = ""
    env: Dict[str, str] = field(default_factory=dict)
    auto_reconnect: bool = True
    reconnect_delay: float = 5.0
    max_reconnect_attempts: int = 5
    health_check_interval: float = 60.0
    request_timeout: float = 30.0
    startup_timeout: float = 15.0
    allowed_tools: Optional[Set[str]] = None  # None = all allowed
    blocked_tools: Set[str] = field(default_factory=set)


class MCPServerLifecycle:
    """Manages the lifecycle of an MCP server connection."""

    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.health = ServerHealth()
        self._reconnect_attempts = 0
        self._reconnect_task: Optional[asyncio.Task] = None
        self._health_check_task: Optional[asyncio.Task] = None
        self._connect_fn: Optional[Callable] = None
        self._disconnect_fn: Optional[Callable] = None
        self._ping_fn: Optional[Callable] = None

    def set_callbacks(
        self,
        connect_fn: Optional[Callable] = None,
        disconnect_fn: Optional[Callable] = None,
        ping_fn: Optional[Callable] = None,
    ) -> None:
        self._connect_fn = connect_fn
        self._disconnect_fn = disconnect_fn
        self._ping_fn = ping_fn

    async def start(self) -> bool:
        """Start the server and begin health monitoring."""
        success = await self._do_connect()
        if success:
            self._start_health_check()
        return success

    async def stop(self) -> None:
        """Stop the server and cancel monitoring."""
        if self._health_check_task:
            self._health_check_task.cancel()
        if self._reconnect_task:
            self._reconnect_task.cancel()
        if self._disconnect_fn:
            try:
                result = self._disconnect_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug("stop: suppressed %s", exc)
        self.health.connected = False

    async def _do_connect(self) -> bool:
        """Attempt to connect."""
        if not self._connect_fn:
            return False
        try:
            result = self._connect_fn()
            if hasattr(result, "__await__"):
                result = await result
            if result:
                self.health.connected = True
                self.health.uptime_start = time.monotonic()
                self.health.consecutive_failures = 0
                self._reconnect_attempts = 0
                logger.info("MCP server '%s' connected", self.config.name)
                return True
            return False
        except Exception as e:
            logger.error("MCP server '%s' connect failed: %s", self.config.name, e)
            return False

    def _start_health_check(self) -> None:
        """Start periodic health checking."""
        if self._health_check_task:
            self._health_check_task.cancel()
        self._health_check_task = asyncio.ensure_future(self._health_check_loop())

    async def _health_check_loop(self) -> None:
        """Periodic health check."""
        while True:
            try:
                await asyncio.sleep(self.config.health_check_interval)
                if not self.health.connected:
                    continue

                if self._ping_fn:
                    self.health.last_ping_at = time.monotonic()
                    try:
                        result = self._ping_fn()
                        if hasattr(result, "__await__"):
                            await result
                        self.health.last_pong_at = time.monotonic()
                        self.health.record_success()
                    except Exception:
                        self.health.record_failure()
                        if not self.health.is_healthy:
                            await self._try_reconnect()

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.warning("Health check error for '%s': %s", self.config.name, e)

    async def _try_reconnect(self) -> None:
        """Attempt to reconnect with backoff."""
        if not self.config.auto_reconnect:
            return
        if self._reconnect_attempts >= self.config.max_reconnect_attempts:
            logger.error(
                "MCP server '%s' max reconnect attempts (%d) reached",
                self.config.name, self.config.max_reconnect_attempts,
            )
            return

        self._reconnect_attempts += 1
        delay = self.config.reconnect_delay * (2 ** (self._reconnect_attempts - 1))
        delay = min(delay, 60)  # Cap at 60s

        logger.info(
            "MCP server '%s' reconnecting in %.1fs (attempt %d/%d)",
            self.config.name, delay, self._reconnect_attempts, self.config.max_reconnect_attempts,
        )

        await asyncio.sleep(delay)

        if self._disconnect_fn:
            try:
                result = self._disconnect_fn()
                if hasattr(result, "__await__"):
                    await result
            except Exception as exc:
                logger.debug("_try_reconnect: suppressed %s", exc)

        success = await self._do_connect()
        if success:
            logger.info("MCP server '%s' reconnected", self.config.name)
        else:
            if self._reconnect_attempts < self.config.max_reconnect_attempts:
                await self._try_reconnect()

    def is_tool_allowed(self, tool_name: str) -> bool:
        """Check if a tool is allowed by this server's config."""
        if tool_name in self.config.blocked_tools:
            return False
        if self.config.allowed_tools is not None:
            return tool_name in self.config.allowed_tools
        return True

    def get_safe_env(self) -> Dict[str, str]:
        """Get safe environment for this server."""
        return build_safe_env(self.config.env)
