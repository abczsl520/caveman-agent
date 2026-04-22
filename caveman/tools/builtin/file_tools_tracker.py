"""File tools tracker — guards, read tracking, dedup, staleness detection.

Extracted from file_tools.py to keep modules under 450 lines.
"""
from __future__ import annotations

import errno
import os
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "is_binary_file",
    "is_blocked_device",
    "check_sensitive_path",
    "get_read_files_summary",
    "clear_read_tracker",
    "reset_file_dedup",
    "notify_other_tool_call",
]


# ── Constants ──────────────────────────────────────────────────────────────

_EXPECTED_WRITE_ERRNOS = {errno.EACCES, errno.EPERM, errno.EROFS}

_BLOCKED_DEVICE_PATHS = frozenset({
    "/dev/zero", "/dev/random", "/dev/urandom", "/dev/full",
    "/dev/stdin", "/dev/tty", "/dev/console",
    "/dev/stdout", "/dev/stderr",
    "/dev/fd/0", "/dev/fd/1", "/dev/fd/2",
})

_SENSITIVE_PATH_PREFIXES = ("/etc/", "/boot/", "/usr/lib/systemd/")
_SENSITIVE_EXACT_PATHS = {"/var/run/docker.sock", "/run/docker.sock"}

_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".ogg", ".flac", ".avi", ".mkv", ".mov",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".wasm",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".db", ".sqlite", ".sqlite3",
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
})

# ── Guards ─────────────────────────────────────────────────────────────────

def is_binary_file(path: str) -> bool:
    """Check if file has a known binary extension."""
    return Path(path).suffix.lower() in _BINARY_EXTENSIONS


def is_blocked_device(path: str) -> bool:
    """Return True if path would hang the process (infinite output / blocking input)."""
    normalized = os.path.expanduser(path)
    if normalized in _BLOCKED_DEVICE_PATHS:
        return True
    if normalized.startswith("/proc/") and normalized.endswith(("/fd/0", "/fd/1", "/fd/2")):
        return True
    return False


def check_sensitive_path(filepath: str) -> Optional[str]:
    """Return error message if path targets a sensitive system location."""
    try:
        resolved = os.path.realpath(os.path.expanduser(filepath))
    except (OSError, ValueError):
        resolved = filepath
    for prefix in _SENSITIVE_PATH_PREFIXES:
        if resolved.startswith(prefix):
            return (
                f"Refusing to write to sensitive system path: {filepath}\n"
                "Use the terminal tool with sudo if you need to modify system files."
            )
    if resolved in _SENSITIVE_EXACT_PATHS:
        return (
            f"Refusing to write to sensitive system path: {filepath}\n"
            "Use the terminal tool with sudo if you need to modify system files."
        )
    return None


def _is_expected_write_exception(exc: Exception) -> bool:
    """Return True for expected write denials (permission errors)."""
    if isinstance(exc, PermissionError):
        return True
    if isinstance(exc, OSError) and exc.errno in _EXPECTED_WRITE_ERRNOS:
        return True
    return False


# ── Read Tracker (per-task dedup + staleness + loop detection) ─────────────

_read_tracker_lock = threading.Lock()
_read_tracker: Dict[str, Dict[str, Any]] = {}


def _get_task_data(task_id: str) -> Dict[str, Any]:
    """Get or create tracker data for a task. Must hold _read_tracker_lock."""
    if task_id not in _read_tracker:
        _read_tracker[task_id] = {
            "last_key": None,
            "consecutive": 0,
            "read_history": set(),
            "dedup": {},
            "read_timestamps": {},
        }
    return _read_tracker[task_id]


def get_read_files_summary(task_id: str = "default") -> List[Dict[str, Any]]:
    """Return list of files read in this session for context compression."""
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id, {})
        read_history = task_data.get("read_history", set())
        seen_paths: Dict[str, List[str]] = {}
        for (path, offset, limit) in read_history:
            if path not in seen_paths:
                seen_paths[path] = []
            seen_paths[path].append(f"lines {offset}-{offset + limit - 1}")
        return [
            {"path": p, "regions": regions}
            for p, regions in sorted(seen_paths.items())
        ]


def clear_read_tracker(task_id: Optional[str] = None) -> None:
    """Clear read tracker. Call on session destroy to prevent memory leaks."""
    with _read_tracker_lock:
        if task_id:
            _read_tracker.pop(task_id, None)
        else:
            _read_tracker.clear()


def reset_file_dedup(task_id: Optional[str] = None) -> None:
    """Clear dedup cache after context compression.

    Original read content is summarised away, so the model needs full
    content if it reads the same file again.
    """
    with _read_tracker_lock:
        if task_id:
            task_data = _read_tracker.get(task_id)
            if task_data and "dedup" in task_data:
                task_data["dedup"].clear()
        else:
            for task_data in _read_tracker.values():
                if "dedup" in task_data:
                    task_data["dedup"].clear()


def notify_other_tool_call(task_id: str = "default") -> None:
    """Reset consecutive counter when a non-read/search tool is called.

    Ensures we only warn/block on truly consecutive repeated reads.
    """
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data:
            task_data["last_key"] = None
            task_data["consecutive"] = 0


def _update_read_timestamp(filepath: str, task_id: str) -> None:
    """Record file mtime after successful write to prevent false staleness."""
    try:
        resolved = str(Path(filepath).expanduser().resolve())
        current_mtime = os.path.getmtime(resolved)
    except (OSError, ValueError):
        return
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if task_data is not None:
            task_data.setdefault("read_timestamps", {})[resolved] = current_mtime


def _check_file_staleness(filepath: str, task_id: str) -> Optional[str]:
    """Check if file was modified since agent last read it.

    Returns warning string if stale, None if fresh or never read.
    """
    try:
        resolved = str(Path(filepath).expanduser().resolve())
    except (OSError, ValueError):
        return None
    with _read_tracker_lock:
        task_data = _read_tracker.get(task_id)
        if not task_data:
            return None
        read_mtime = task_data.get("read_timestamps", {}).get(resolved)
    if read_mtime is None:
        return None
    try:
        current_mtime = os.path.getmtime(resolved)
    except OSError:
        return None
    if current_mtime != read_mtime:
        return (
            f"Warning: {filepath} was modified since you last read it "
            "(external edit or concurrent agent). Consider re-reading."
        )
    return None


