"""Debug session infrastructure for tools.

Provides a lightweight debug logger that records tool calls to a JSON log file.
Activated by a tool-specific environment variable. When disabled, all methods
are cheap no-ops.
"""
from __future__ import annotations

import json
import logging
import os
import time
import uuid
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)


class DebugSession:
    """Per-tool debug session that records tool calls to a JSON log file."""

    def __init__(self, tool_name: str, env_var: str = "") -> None:
        self._tool_name = tool_name
        self._env_var = env_var or f"{tool_name.upper()}_DEBUG"
        self._session_id = str(uuid.uuid4())[:12]
        self._calls: list[dict[str, Any]] = []
        self._start_time = time.time()

    @property
    def enabled(self) -> bool:
        return os.getenv(self._env_var, "").lower() in ("1", "true", "yes")

    def log_call(self, function_name: str, data: dict[str, Any] | None = None) -> None:
        """Log a tool call. No-op when debug mode is off."""
        if not self.enabled:
            return
        self._calls.append({
            "function": function_name,
            "timestamp": time.time(),
            "data": data or {},
        })

    def save(self) -> Path | None:
        """Save the debug log to disk. Returns path or None if disabled."""
        if not self.enabled or not self._calls:
            return None

        debug_dir = CAVEMAN_HOME / "debug"
        debug_dir.mkdir(parents=True, exist_ok=True)
        path = debug_dir / f"{self._tool_name}_{self._session_id}.json"

        data = {
            "tool": self._tool_name,
            "session_id": self._session_id,
            "start_time": self._start_time,
            "duration": time.time() - self._start_time,
            "calls": self._calls,
        }
        path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        return path

    def get_session_info(self) -> dict[str, Any]:
        """Get debug session info for external callers."""
        return {
            "tool": self._tool_name,
            "session_id": self._session_id,
            "enabled": self.enabled,
            "call_count": len(self._calls),
            "duration": time.time() - self._start_time,
        }
