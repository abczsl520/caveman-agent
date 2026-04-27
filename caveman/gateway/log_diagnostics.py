"""Current-startup gateway log diagnostics.

This module intentionally scans only the current gateway process window. The
primary boundary is the latest PID-file startup marker (`PID file written ...
PID N`), with an ISO `started_at` timestamp fallback for logs that predate the
marker.
"""
from __future__ import annotations

import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

DEFAULT_PATTERNS = (
    "ERROR",
    "Traceback",
    "Permission DENIED",
    "no such column",
    "ASK mode",
    "hot-reload",
    "Agent inactive",
    "Unknown gateway",
)
_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2})[ T](\d{2}:\d{2}:\d{2})")


def _default_pidfile() -> Path:
    return CAVEMAN_HOME / "gateway.pid"


def _default_logfile() -> Path:
    return CAVEMAN_HOME / "logs" / "gateway.log"


def _read_pid_record(pidfile: Path) -> dict[str, Any] | None:
    try:
        data = json.loads(pidfile.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def _parse_iso(value: Any) -> datetime | None:
    if not isinstance(value, str) or not value:
        return None
    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return None


def _parse_log_time(line: str, tz=timezone.utc) -> datetime | None:
    match = _TS_RE.match(line)
    if not match:
        return None
    try:
        dt = datetime.strptime(" ".join(match.groups()), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        return None
    return dt.replace(tzinfo=tz)


def _find_start_index(lines: list[str], record: dict[str, Any] | None) -> tuple[int, bool, str]:
    if not record:
        return 0, False, "missing_or_invalid_pidfile"
    pid = record.get("pid")
    if pid is not None:
        marker = "PID file written:"
        pid_text = f"PID {pid}"
        for idx in range(len(lines) - 1, -1, -1):
            line = lines[idx]
            if marker in line and pid_text in line:
                return idx, True, "pid_marker"
    started_at = _parse_iso(record.get("started_at"))
    if started_at:
        local_started = started_at.astimezone()
        for idx, line in enumerate(lines):
            ts = _parse_log_time(line, tz=local_started.tzinfo or timezone.utc)
            if ts and ts >= local_started.replace(microsecond=0):
                return idx, True, "started_at"
    return 0, False, "no_start_boundary_found"


def scan_current_startup_log(
    *,
    pidfile: Path | None = None,
    logfile: Path | None = None,
    expected_pid: int | None = None,
    patterns: tuple[str, ...] = DEFAULT_PATTERNS,
    sample_limit: int = 3,
) -> dict[str, Any]:
    """Scan gateway logs bounded to the current process startup window.

    When ``expected_pid`` is provided by a caller that has already verified the
    gateway process is running, the pidfile must match that PID before any
    bounded window or healthy marker can be trusted.
    """
    pidfile = pidfile or _default_pidfile()
    logfile = logfile or _default_logfile()
    try:
        lines = logfile.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        lines = []
    record = _read_pid_record(pidfile)
    if expected_pid is not None and (not record or record.get("pid") != expected_pid):
        record = None
        start_idx, bounded, boundary = 0, False, "pid_mismatch"
    else:
        start_idx, bounded, boundary = _find_start_index(lines, record)
    window = lines[start_idx:]

    pattern_report: dict[str, dict[str, Any]] = {}
    for pattern in patterns:
        hits = [line for line in window if pattern in line]
        pattern_report[pattern] = {"count": len(hits), "samples": hits[-sample_limit:]}

    return {
        "bounded": bounded,
        "boundary": boundary,
        "pid": record.get("pid") if record else None,
        "started_at": record.get("started_at") if record else None,
        "startup_line_index": start_idx,
        "line_count": len(window),
        "patterns": pattern_report,
        "healthy_markers": {
            "discord_connected": bounded and any("Discord connected:" in line for line in window),
            "slash_commands_synced": bounded and any("Synced " in line and "slash commands" in line for line in window),
            "health_started": bounded and any("Health check server on port" in line for line in window),
        },
    }


__all__ = ["DEFAULT_PATTERNS", "scan_current_startup_log"]
