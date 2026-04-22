"""File tools — full-depth port from Hermes file_tools.py.

Features ported:
- Device path blocklist (infinite output / blocking input)
- Sensitive path protection (/etc/, /boot/, docker.sock)
- Read dedup (skip re-reads of unchanged files)
- Staleness detection (warn on write if file changed since last read)
- Consecutive read/search loop detection (warn at 3, block at 4)
- Character-count guard (reject oversized reads)
- Binary file guard (by extension)
- Read tracker with per-task isolation
- Secret redaction on read output
"""
from __future__ import annotations


import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

# ── Constants (local only — shared constants in file_tools_tracker.py) ─────

_DEFAULT_MAX_READ_CHARS = 100_000
_LARGE_FILE_HINT_BYTES = 512_000

# ── Config cache ───────────────────────────────────────────────────────────

_max_read_chars_cached: Optional[int] = None


def _get_max_read_chars() -> int:
    """Return configured max chars per read, cached after first call."""
    global _max_read_chars_cached
    if _max_read_chars_cached is not None:
        return _max_read_chars_cached
    try:
        from caveman.config.loader import load_config
        cfg = load_config()
        val = cfg.get("file_read_max_chars")
        if isinstance(val, (int, float)) and val > 0:
            _max_read_chars_cached = int(val)
            return _max_read_chars_cached
    except Exception as exc:
        logger.debug("_get_max_read_chars: suppressed %s", exc)
    _max_read_chars_cached = _DEFAULT_MAX_READ_CHARS
    return _max_read_chars_cached



# Import guards and tracker from split module

from caveman.tools.builtin.file_tools_search import search_in_file, file_diff  # noqa: F401
from caveman.tools.builtin.file_tools_tracker import (
    is_binary_file, is_blocked_device, check_sensitive_path,
    _is_expected_write_exception,
    get_read_files_summary, clear_read_tracker, reset_file_dedup,  # noqa: F401
    notify_other_tool_call,  # noqa: F401
    _update_read_timestamp,
    _check_file_staleness, _get_task_data, _read_tracker_lock,
)

__all__ = ["read_file", "replace_in_file", "create_file", "write_file", "patch_file"]


# ── Redaction ──────────────────────────────────────────────────────────────

def _redact_content(text: str) -> str:
    """Redact sensitive patterns from file content."""
    try:
        from caveman.security.redact import redact_text
        return redact_text(text)
    except ImportError:
        return text


# ── Core File Operations ──────────────────────────────────────────────────

def read_file(
    path: str,
    offset: int = 1,
    limit: int = 500,
    *,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Read a text file with line numbers, pagination, dedup, and guards."""
    # Device path guard
    if is_blocked_device(path):
        return {"error": f"Cannot read '{path}': device file that would block or produce infinite output."}

    resolved = Path(path).expanduser().resolve()
    resolved_str = str(resolved)

    # Binary file guard
    if is_binary_file(str(resolved)):
        ext = resolved.suffix.lower()
        return {"error": f"Binary file cannot be read: '{path}' ({ext}). Use vision_analyze for images."}

    # Dedup check — skip re-reads of unchanged files
    dedup_key = (resolved_str, offset, limit)
    with _read_tracker_lock:
        task_data = _get_task_data(task_id)
        cached_mtime = task_data.get("dedup", {}).get(dedup_key)

    if cached_mtime is not None:
        try:
            current_mtime = os.path.getmtime(resolved_str)
            if current_mtime == cached_mtime:
                return {
                    "content": "File unchanged since last read. Refer to earlier read_file result.",
                    "path": path, "dedup": True,
                }
        except OSError:
            pass  # intentional: OSError suppressed

    # Perform the read
    if not resolved.exists():
        # Suggest similar files
        parent = resolved.parent
        suggestions = []
        if parent.exists():
            name_lower = resolved.name.lower()
            for f in parent.iterdir():
                if name_lower in f.name.lower() or f.stem.lower() in name_lower:
                    suggestions.append(str(f.name))
                    if len(suggestions) >= 5:
                        break
        result: Dict[str, Any] = {"error": f"File not found: {path}"}
        if suggestions:
            result["suggestions"] = suggestions
        return result

    try:
        content = resolved.read_text(encoding="utf-8", errors="replace")
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except OSError as e:
        return {"error": f"Cannot read {path}: {e}"}

    lines = content.splitlines(keepends=True)
    total_lines = len(lines)
    file_size = resolved.stat().st_size

    # Apply offset/limit
    start = max(0, offset - 1)
    end = start + limit
    selected = lines[start:end]
    truncated = end < total_lines

    # Format with line numbers
    numbered = "".join(
        f"{start + i + 1}|{line}" for i, line in enumerate(selected)
    )

    # Character-count guard
    max_chars = _get_max_read_chars()
    if len(numbered) > max_chars:
        return {
            "error": f"Read produced {len(numbered):,} chars exceeding limit ({max_chars:,}). Use offset and limit.",
            "path": path, "total_lines": total_lines, "file_size": file_size,
        }

    # Redact secrets
    numbered = _redact_content(numbered)

    result = {
        "content": numbered,
        "path": path,
        "total_lines": total_lines,
        "file_size": file_size,
        "lines_shown": f"{start + 1}-{start + len(selected)}",
    }
    if truncated:
        result["truncated"] = True

    # Large file hint
    if file_size > _LARGE_FILE_HINT_BYTES and limit > 200 and truncated:
        result["_hint"] = (
            f"This file is large ({file_size:,} bytes). "
            "Consider reading only the section you need with offset and limit."
        )

    # Track for consecutive-loop detection
    read_key = ("read", path, offset, limit)
    with _read_tracker_lock:
        task_data = _get_task_data(task_id)
        task_data["read_history"].add((path, offset, limit))
        if task_data["last_key"] == read_key:
            task_data["consecutive"] += 1
        else:
            task_data["last_key"] = read_key
            task_data["consecutive"] = 1
        count = task_data["consecutive"]
        # Store mtime for dedup + staleness
        try:
            mtime_now = os.path.getmtime(resolved_str)
            task_data["dedup"][dedup_key] = mtime_now
            task_data["read_timestamps"][resolved_str] = mtime_now
        except OSError:
            pass  # intentional: OSError suppressed

    if count >= 4:
        return {
            "error": f"BLOCKED: You have read this exact file region {count} times in a row. "
                     "The content has NOT changed. STOP re-reading and proceed.",
            "path": path, "already_read": count,
        }
    elif count >= 3:
        result["_warning"] = (
            f"You have read this exact file region {count} times consecutively. "
            "The content has not changed. Use the information you already have."
        )

    return result


def replace_in_file(
    path: str,
    old_string: str,
    new_string: str,
    *,
    replace_all: bool = False,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Replace text in a file with staleness detection and sensitive path check."""
    sensitive_err = check_sensitive_path(path)
    if sensitive_err:
        return {"error": sensitive_err}

    stale_warning = _check_file_staleness(path, task_id)
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, PermissionError) as e:
        return {"error": str(e)}

    if old_string not in content:
        return {
            "error": f"Could not find the specified text in {path}",
            "_hint": "Use read_file to verify the current content, or search_files to locate the text.",
        }

    occurrences = content.count(old_string)
    if occurrences > 1 and not replace_all:
        return {
            "error": f"Found {occurrences} occurrences of old_string in {path}. "
                     "Include more surrounding context to ensure uniqueness, or set replace_all=true.",
        }

    new_content = content.replace(old_string, new_string) if replace_all else content.replace(old_string, new_string, 1)

    try:
        resolved.write_text(new_content, encoding="utf-8")
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except OSError as e:
        if _is_expected_write_exception(e):
            return {"error": f"Write denied: {e}"}
        logger.error("replace_in_file error: %s", e, exc_info=True)
        return {"error": str(e)}

    _update_read_timestamp(path, task_id)

    result: Dict[str, Any] = {
        "success": True, "path": path,
        "replacements": occurrences if replace_all else 1,
    }
    if stale_warning:
        result["_warning"] = stale_warning
    return result


def create_file(path: str, content: str = "", *, parents: bool = True) -> Dict[str, Any]:
    """Create a new file with sensitive path check."""
    sensitive_err = check_sensitive_path(path)
    if sensitive_err:
        return {"error": sensitive_err}

    resolved = Path(path).expanduser().resolve()
    if resolved.exists():
        return {"error": f"File already exists: {path}. Use replace_in_file or write to overwrite."}

    try:
        if parents:
            resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except OSError as e:
        return {"error": str(e)}

    return {"success": True, "path": path, "size": len(content)}


def write_file(
    path: str,
    content: str,
    *,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Write content to a file (overwrite) with staleness detection."""
    sensitive_err = check_sensitive_path(path)
    if sensitive_err:
        return {"error": sensitive_err}

    stale_warning = _check_file_staleness(path, task_id)
    resolved = Path(path).expanduser().resolve()

    try:
        resolved.parent.mkdir(parents=True, exist_ok=True)
        resolved.write_text(content, encoding="utf-8")
    except PermissionError:
        return {"error": f"Permission denied: {path}"}
    except OSError as e:
        if _is_expected_write_exception(e):
            logger.debug("write_file expected denial: %s", e)
        else:
            logger.error("write_file error: %s", e, exc_info=True)
        return {"error": str(e)}

    _update_read_timestamp(path, task_id)

    result: Dict[str, Any] = {"success": True, "path": path, "size": len(content)}
    if stale_warning:
        result["_warning"] = stale_warning
    return result


def patch_file(
    path: str,
    patches: List[Dict[str, str]],
    *,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Apply multiple patches to a file with staleness detection."""
    sensitive_err = check_sensitive_path(path)
    if sensitive_err:
        return {"error": sensitive_err}

    stale_warning = _check_file_staleness(path, task_id)
    resolved = Path(path).expanduser().resolve()

    if not resolved.exists():
        return {"error": f"File not found: {path}"}

    try:
        content = resolved.read_text(encoding="utf-8")
    except (OSError, PermissionError) as e:
        return {"error": str(e)}

    applied = 0
    errors: List[str] = []
    for i, patch in enumerate(patches):
        old = patch.get("old", "")
        new = patch.get("new", "")
        if not old:
            errors.append(f"Patch {i}: missing 'old' field")
            continue
        if old not in content:
            errors.append(f"Patch {i}: old text not found")
            continue
        content = content.replace(old, new, 1)
        applied += 1

    if applied > 0:
        try:
            resolved.write_text(content, encoding="utf-8")
            _update_read_timestamp(path, task_id)
        except (OSError, PermissionError) as e:
            return {"error": str(e)}

    result: Dict[str, Any] = {
        "success": applied > 0, "path": path,
        "applied": applied, "total": len(patches),
    }
    if errors:
        result["errors"] = errors
    if stale_warning:
        result["_warning"] = stale_warning
    return result


