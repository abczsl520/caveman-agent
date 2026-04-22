"""Gateway graceful restart — SIGUSR1-based restart with drain support.

Restart flow:
1. /restart command or SIGUSR1 signal received
2. Set _restart_requested flag
3. Drain active agents (wait for completion, up to timeout)
4. Exit with code 75 (EX_TEMPFAIL) — tells service manager to restart
5. Service manager (launchd/systemd) starts a new process

Inspired by Hermes gateway/restart.py + gateway/run.py and
OpenClaw src/infra/restart.ts.
"""
from __future__ import annotations

import asyncio
import logging
import os
import signal
import sys
from pathlib import Path
from typing import Optional

__all__ = [
    "RESTART_EXIT_CODE",
    "DEFAULT_DRAIN_TIMEOUT",
    "RESTART_COOLDOWN_S",
    "parse_drain_timeout",
    "request_restart_via_signal",
    "request_restart_via_launchd",
    "request_restart_via_systemd",
    "trigger_restart",
    "write_restart_sentinel",
    "consume_restart_sentinel",
]


logger = logging.getLogger("caveman.gateway.restart")

# EX_TEMPFAIL from sysexits.h — used to ask the service manager to restart.
# launchd KeepAlive.SuccessfulExit=false means non-zero exit → auto-restart.
RESTART_EXIT_CODE = 75

# Default drain timeout: wait this long for active agents to finish.
DEFAULT_DRAIN_TIMEOUT = 300.0  # 5 minutes (matches OpenClaw's default)

# Minimum cooldown between restart requests.
RESTART_COOLDOWN_S = 30.0

_last_restart_at: float = 0.0


def parse_drain_timeout(raw: object) -> float:
    """Parse a configured drain timeout, falling back to default."""
    try:
        value = float(raw) if str(raw or "").strip() else DEFAULT_DRAIN_TIMEOUT
    except (TypeError, ValueError):
        return DEFAULT_DRAIN_TIMEOUT
    return max(0.0, value)


def request_restart_via_signal(pid: Optional[int] = None) -> bool:
    """Send SIGUSR1 to request a graceful restart.

    Args:
        pid: Target PID. Defaults to current process.

    Returns:
        True if signal was sent successfully.
    """
    if not hasattr(signal, "SIGUSR1"):
        logger.warning("SIGUSR1 not available on this platform")
        return False

    target = pid or os.getpid()
    try:
        os.kill(target, signal.SIGUSR1)
        logger.info("Sent SIGUSR1 to PID %d", target)
        return True
    except (ProcessLookupError, PermissionError, OSError) as e:
        logger.error("Failed to send SIGUSR1 to PID %d: %s", target, e)
        return False


async def request_restart_via_launchd() -> bool:
    """Request restart through launchd (macOS).

    Uses `launchctl kickstart -k` which kills the old process and
    starts a new one atomically. Async to avoid blocking the event loop.
    """
    if sys.platform != "darwin":
        return False

    label = _get_launchd_label()
    uid = os.getuid()
    target = f"gui/{uid}/{label}"

    try:
        proc = await asyncio.wait_for(
            asyncio.create_subprocess_exec(
                "launchctl", "kickstart", "-k", target,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            ),
            timeout=10,
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0:
            logger.info("launchd restart requested: %s", target)
            return True
        logger.warning("launchctl kickstart failed: %s", stderr.decode().strip())
    except (asyncio.TimeoutError, FileNotFoundError, OSError) as e:
        logger.warning("launchctl kickstart error: %s", e)

    return False


async def request_restart_via_systemd() -> bool:
    """Request restart through systemd (Linux). Async to avoid blocking."""
    if sys.platform != "linux":
        return False

    service = _get_systemd_service()
    for args in [["systemctl", "--user", "restart", service],
                 ["systemctl", "restart", service]]:
        try:
            proc = await asyncio.wait_for(
                asyncio.create_subprocess_exec(
                    *args,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                ),
                timeout=10,
            )
            await proc.communicate()
            if proc.returncode == 0:
                logger.info("systemd restart requested: %s", " ".join(args))
                return True
        except (asyncio.TimeoutError, FileNotFoundError, OSError):
            continue

    return False


async def trigger_restart() -> dict:
    """Trigger a gateway restart using the best available method.

    Tries in order: launchd (macOS) → systemd (Linux) → SIGUSR1 (fallback).
    Async to avoid blocking the event loop during subprocess calls.

    Returns:
        dict with 'ok', 'method', and optional 'detail'.
    """
    global _last_restart_at
    import time

    now = time.monotonic()
    if now - _last_restart_at < RESTART_COOLDOWN_S:
        return {"ok": False, "method": "cooldown",
                "detail": f"Restart cooldown ({RESTART_COOLDOWN_S}s). Try again later."}

    _last_restart_at = now

    # Try platform-specific service manager first
    if sys.platform == "darwin":
        if await request_restart_via_launchd():
            return {"ok": True, "method": "launchd"}
    elif sys.platform == "linux":
        if await request_restart_via_systemd():
            return {"ok": True, "method": "systemd"}

    # Fallback: SIGUSR1 to self (sync, but instant — no subprocess)
    if request_restart_via_signal():
        return {"ok": True, "method": "sigusr1"}

    return {"ok": False, "method": "none", "detail": "No restart method available"}


# ── Restart sentinel ──

def write_restart_sentinel(
    *,
    kind: str = "restart",
    reason: Optional[str] = None,
    session_key: Optional[str] = None,
) -> Path:
    """Write a restart sentinel file so the new process knows why it restarted.

    The new process reads and deletes this file on startup.
    """
    from caveman.paths import CAVEMAN_HOME
    sentinel_path = CAVEMAN_HOME / "restart-sentinel.json"

    import json
    data = {
        "version": 1,
        "kind": kind,
        "reason": reason,
        "session_key": session_key,
        "ts": _utc_now_iso(),
        "pid": os.getpid(),
    }
    sentinel_path.parent.mkdir(parents=True, exist_ok=True)
    # Atomic write: temp file + rename (same pattern as status.py)
    import tempfile
    content = json.dumps(data, indent=2) + "\n"
    fd, tmp = tempfile.mkstemp(dir=sentinel_path.parent, suffix=".tmp")
    closed = False
    try:
        os.write(fd, content.encode("utf-8"))
        os.fsync(fd)
        os.close(fd)
        closed = True
        os.replace(tmp, str(sentinel_path))
    except Exception:
        if not closed:
            try:
                os.close(fd)
            except OSError:
                pass  # intentional: OSError suppressed
        try:
            os.unlink(tmp)
        except OSError:
            pass  # intentional: OSError suppressed
        raise
    logger.info("Restart sentinel written: %s", sentinel_path)
    return sentinel_path


def consume_restart_sentinel() -> Optional[dict]:
    """Read and delete the restart sentinel file.

    Returns the sentinel data if it existed, None otherwise.
    """
    from caveman.paths import CAVEMAN_HOME
    import json

    sentinel_path = CAVEMAN_HOME / "restart-sentinel.json"
    if not sentinel_path.exists():
        return None

    try:
        data = json.loads(sentinel_path.read_text(encoding="utf-8"))
        sentinel_path.unlink(missing_ok=True)
        logger.info("Consumed restart sentinel: kind=%s reason=%s",
                     data.get("kind"), data.get("reason"))
        return data
    except (json.JSONDecodeError, OSError) as e:
        logger.warning("Failed to read restart sentinel: %s", e)
        try:
            sentinel_path.unlink(missing_ok=True)
        except OSError:
            pass  # intentional: OSError suppressed
        return None


# ── Helpers ──

def _utc_now_iso() -> str:
    from datetime import datetime, timezone
    return datetime.now(timezone.utc).isoformat()


def _get_launchd_label() -> str:
    return os.environ.get("CAVEMAN_LAUNCHD_LABEL", "ai.caveman.gateway")


def _get_systemd_service() -> str:
    return os.environ.get("CAVEMAN_SYSTEMD_SERVICE", "caveman-gateway.service")
