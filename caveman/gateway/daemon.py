"""Daemon Manager — background process lifecycle management.

Manages the caveman daemon process (start/stop/restart/status).
Core patterns from OpenClaw src/daemon/ (10K LOC — extracted essentials).
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

__all__ = [
    "DaemonStatus",
    "get_status",
    "start",
    "stop",
    "restart",
]


logger = logging.getLogger("caveman.gateway.daemon")

_PID_FILE = Path.home() / ".caveman" / "daemon.pid"
_LOG_FILE = Path.home() / ".caveman" / "daemon.log"
_STATUS_FILE = Path.home() / ".caveman" / "daemon.status"


@dataclass
class DaemonStatus:
    """Status of the daemon process."""
    running: bool = False
    pid: int = 0
    uptime_seconds: float = 0
    started_at: float = 0
    version: str = ""
    sessions_active: int = 0
    memory_mb: float = 0


def get_status() -> DaemonStatus:
    """Get current daemon status."""
    status = DaemonStatus()

    if not _PID_FILE.exists():
        return status

    try:
        pid = int(_PID_FILE.read_text(encoding="utf-8").strip())
    except (ValueError, OSError):
        return status

    # Check if process is alive
    try:
        os.kill(pid, 0)
        status.running = True
        status.pid = pid
    except (OSError, ProcessLookupError):
        # Stale PID file
        _PID_FILE.unlink(missing_ok=True)
        return status

    # Read status file for details
    if _STATUS_FILE.exists():
        try:
            data = json.loads(_STATUS_FILE.read_text(encoding="utf-8"))
            status.started_at = data.get("started_at", 0)
            status.version = data.get("version", "")
            status.sessions_active = data.get("sessions_active", 0)
            if status.started_at:
                status.uptime_seconds = time.time() - status.started_at
        except Exception as exc:
            logger.debug("get_status: suppressed %s", exc)

    # Get memory usage
    try:
        result = subprocess.run(
            ["ps", "-o", "rss=", "-p", str(pid)],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0:
            rss_kb = int(result.stdout.strip())
            status.memory_mb = rss_kb / 1024
    except Exception as exc:
        logger.debug("get_status: suppressed %s", exc)

    return status


def start(command: str = "", env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Start the daemon."""
    status = get_status()
    if status.running:
        return {"success": False, "error": f"Already running (PID {status.pid})"}

    cmd = command or "caveman serve"
    full_env = {**os.environ, **(env or {})}

    try:
        proc = subprocess.Popen(
            cmd.split(),
            stdout=open(_LOG_FILE, "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
            env=full_env,
            start_new_session=True,
        )

        # Write PID file
        _PID_FILE.parent.mkdir(parents=True, exist_ok=True)
        _PID_FILE.write_text(str(proc.pid), encoding="utf-8")

        # Write status
        _STATUS_FILE.write_text(json.dumps({
            "started_at": time.time(),
            "pid": proc.pid,
            "command": cmd,
        }, ensure_ascii=False), encoding="utf-8")

        return {"success": True, "pid": proc.pid}

    except Exception as e:
        return {"success": False, "error": str(e)}


def stop(graceful_timeout: float = 10.0) -> Dict[str, Any]:
    """Stop the daemon."""
    status = get_status()
    if not status.running:
        return {"success": True, "message": "Not running"}

    try:
        # Graceful shutdown
        os.kill(status.pid, signal.SIGTERM)

        # Wait for graceful shutdown
        deadline = time.time() + graceful_timeout
        while time.time() < deadline:
            try:
                os.kill(status.pid, 0)
                time.sleep(0.5)
            except (OSError, ProcessLookupError):
                break
        else:
            # Force kill
            try:
                os.kill(status.pid, signal.SIGKILL)
            except (OSError, ProcessLookupError):
                pass  # intentional: OSError/ProcessLookupError suppressed

        # Cleanup
        _PID_FILE.unlink(missing_ok=True)
        _STATUS_FILE.unlink(missing_ok=True)

        return {"success": True, "pid": status.pid}

    except Exception as e:
        return {"success": False, "error": str(e)}


def restart(command: str = "", env: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Restart the daemon."""
    stop_result = stop()
    if not stop_result.get("success"):
        return stop_result
    time.sleep(1)
    return start(command, env)
