"""Gateway process status — PID file, runtime state, and process identity.

Provides PID-file based detection of whether the gateway is running,
runtime state tracking (starting/running/draining/stopped), and
process identity verification to prevent PID reuse false positives.

Inspired by Hermes gateway/status.py (MIT, Nous Research) and
OpenClaw src/infra/restart.ts.

The PID file lives at ``{CAVEMAN_HOME}/gateway.pid``.
The runtime status file lives at ``{CAVEMAN_HOME}/gateway_state.json``.
"""
from __future__ import annotations

import json
import logging
import os
import signal
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger("caveman.gateway.status")

_GATEWAY_KIND = "caveman-gateway"


# ── Path helpers ──

def _pid_path() -> Path:
    return CAVEMAN_HOME / "gateway.pid"


def _state_path() -> Path:
    return CAVEMAN_HOME / "gateway_state.json"


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# ── JSON file helpers ──

def _read_json(path: Path) -> Optional[dict[str, Any]]:
    if not path.exists():
        return None
    try:
        raw = path.read_text(encoding="utf-8").strip()
        if not raw:
            return None
        data = json.loads(raw)
        return data if isinstance(data, dict) else None
    except (OSError, json.JSONDecodeError):
        return None


def _write_json(path: Path, data: dict[str, Any]) -> None:
    """Atomic JSON write — write to temp file then rename.

    Prevents corrupted files if the process crashes mid-write.
    """
    import tempfile
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2) + "\n"
    # Write to temp file in same directory (same filesystem for atomic rename)
    fd, tmp_path = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp_path, str(path))  # Atomic on POSIX
    except Exception:
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass  # intentional: OSError suppressed
        try:
            os.unlink(tmp_path)
        except OSError:
            pass  # intentional: OSError suppressed
        raise


# ── Process identity ──

def _get_process_cmdline(pid: int) -> Optional[str]:
    """Get the command line of a process (macOS/Linux)."""
    try:
        result = subprocess.run(
            ["ps", "-o", "command=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, OSError):
        pass  # intentional: OSError suppressed
    return None


def _looks_like_gateway(pid: int) -> bool:
    """Check if a PID looks like a Caveman gateway process.

    Intentionally excludes 'caveman flywheel' — the flywheel is a self-audit
    tool, not a gateway. Including it would cause /restart to target the wrong process.
    """
    cmdline = _get_process_cmdline(pid)
    if not cmdline:
        return False
    # Only match gateway-specific patterns, NOT flywheel/audit/bench/etc.
    patterns = ("caveman serve", "caveman gateway", "run_gateway")
    return any(p in cmdline for p in patterns)


def _is_pid_alive(pid: int) -> bool:
    """Check if a PID is alive (signal 0 = existence check)."""
    try:
        os.kill(pid, 0)
        return True
    except (ProcessLookupError, PermissionError):
        return False


# ── PID file management ──

def _build_pid_record() -> dict[str, Any]:
    return {
        "pid": os.getpid(),
        "kind": _GATEWAY_KIND,
        "argv": list(sys.argv),
        "started_at": _utc_now_iso(),
        "python": sys.executable,
    }


def write_pid_file() -> None:
    """Write the current process PID and metadata to the gateway PID file."""
    _write_json(_pid_path(), _build_pid_record())
    logger.info("PID file written: %s (PID %d)", _pid_path(), os.getpid())


def remove_pid_file() -> None:
    """Remove the gateway PID file."""
    try:
        _pid_path().unlink(missing_ok=True)
    except OSError:
        pass  # intentional: OSError suppressed


def get_running_pid() -> Optional[int]:
    """Return the PID of a running gateway, or None.

    Validates:
    1. PID file exists and is valid JSON
    2. Process is alive (signal 0)
    3. Process looks like a gateway (cmdline check)

    Cleans up stale PID files automatically.
    """
    record = _read_json(_pid_path())
    if not record:
        return None

    try:
        pid = int(record["pid"])
    except (KeyError, TypeError, ValueError):
        remove_pid_file()
        return None

    if not _is_pid_alive(pid):
        logger.debug("Stale PID file (process %d dead), cleaning up", pid)
        remove_pid_file()
        return None

    # Verify process identity to prevent PID reuse false positives
    if not _looks_like_gateway(pid):
        logger.debug("PID %d is alive but not a gateway process, cleaning up", pid)
        remove_pid_file()
        return None

    return pid


def is_gateway_running() -> bool:
    """Check if the gateway daemon is currently running."""
    return get_running_pid() is not None


# ── Runtime state ──

def write_runtime_state(
    *,
    state: Optional[str] = None,
    exit_reason: Optional[str] = None,
    restart_requested: Optional[bool] = None,
    active_sessions: Optional[int] = None,
    platform: Optional[str] = None,
    platform_state: Optional[str] = None,
) -> None:
    """Update the persisted gateway runtime state.

    States: starting → running → draining → stopped
    """
    path = _state_path()
    data = _read_json(path) or {
        "pid": os.getpid(),
        "kind": _GATEWAY_KIND,
        "state": "starting",
        "platforms": {},
    }

    data["pid"] = os.getpid()
    data["updated_at"] = _utc_now_iso()

    if state is not None:
        data["state"] = state
    if exit_reason is not None:
        data["exit_reason"] = exit_reason
    if restart_requested is not None:
        data["restart_requested"] = restart_requested
    if active_sessions is not None:
        data["active_sessions"] = max(0, active_sessions)

    if platform is not None:
        platforms = data.setdefault("platforms", {})
        plat_data = platforms.setdefault(platform, {})
        if platform_state is not None:
            plat_data["state"] = platform_state
        plat_data["updated_at"] = _utc_now_iso()

    _write_json(path, data)


def read_runtime_state() -> Optional[dict[str, Any]]:
    """Read the persisted gateway runtime state."""
    return _read_json(_state_path())


def terminate_pid(pid: int, *, force: bool = False) -> None:
    """Terminate a PID with appropriate signal.

    Args:
        pid: Process ID to terminate.
        force: Use SIGKILL instead of SIGTERM.
    """
    sig = signal.SIGKILL if force else signal.SIGTERM
    os.kill(pid, sig)
    logger.info("Sent %s to PID %d", sig.name, pid)

from caveman.gateway.status_depth import (  # noqa: F401,E402  # depth wiring
    TokenStats,
    ModelInfo,
    MODEL_INFO_DB,
    get_model_info,
    SessionListEntry,
    format_session_list,
    format_token_stats,
    format_model_info,
)

__all__ = [
    "write_pid_file",
    "remove_pid_file",
    "get_running_pid",
    "is_gateway_running",
    "write_runtime_state",
    "read_runtime_state",
    "terminate_pid",
    "TokenStats",
    "ModelInfo",
    "MODEL_INFO_DB",
    "get_model_info",
    "SessionListEntry",
    "format_session_list",
    "format_token_stats",
    "format_model_info",
]

