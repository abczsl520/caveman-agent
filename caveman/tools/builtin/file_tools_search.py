"""File tools — search and diff operations.

Extracted from file_tools.py to keep modules under 450 lines.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from caveman.tools.builtin.file_tools_tracker import (
    is_binary_file, _get_task_data, _read_tracker_lock,
)

__all__ = ["search_in_file", "file_diff"]


logger = logging.getLogger(__name__)


def _redact_content(text: str) -> str:
    """Redact sensitive patterns from file content."""
    try:
        from caveman.security.redact import redact_text
        return redact_text(text)
    except ImportError:
        return text


def search_in_file(
    pattern: str,
    path: str = ".",
    *,
    target: str = "content",
    file_glob: Optional[str] = None,
    limit: int = 50,
    offset: int = 0,
    context_lines: int = 0,
    task_id: str = "default",
) -> Dict[str, Any]:
    """Search file contents or find files by name with loop detection."""
    import re as _re

    search_key = ("search", pattern, target, str(path), file_glob or "", limit, offset)
    with _read_tracker_lock:
        task_data = _get_task_data(task_id)
        if task_data["last_key"] == search_key:
            task_data["consecutive"] += 1
        else:
            task_data["last_key"] = search_key
            task_data["consecutive"] = 1
        count = task_data["consecutive"]

    if count >= 4:
        return {
            "error": f"BLOCKED: You have run this exact search {count} times in a row. "
                     "The results have NOT changed. STOP re-searching and proceed.",
            "pattern": pattern, "already_searched": count,
        }

    search_path = Path(path).expanduser().resolve()
    matches: List[Dict[str, Any]] = []

    if target == "files":
        # Find files by glob pattern
        if search_path.is_dir():
            glob_pattern = pattern if "*" in pattern or "?" in pattern else f"*{pattern}*"
            found = sorted(search_path.rglob(glob_pattern), key=lambda p: p.stat().st_mtime if p.exists() else 0, reverse=True)
            for f in found[offset:offset + limit]:
                try:
                    stat = f.stat()
                    matches.append({"path": str(f), "size": stat.st_size, "modified": stat.st_mtime})
                except OSError:
                    matches.append({"path": str(f)})
        result = {"matches": matches, "total_count": len(matches), "pattern": pattern, "target": "files"}
    else:
        # Content search with regex
        try:
            regex = _re.compile(pattern, _re.IGNORECASE)
        except _re.error as e:
            return {"error": f"Invalid regex pattern: {e}"}

        files_to_search: List[Path] = []
        if search_path.is_file():
            files_to_search = [search_path]
        elif search_path.is_dir():
            glob = file_glob or "*"
            files_to_search = [f for f in search_path.rglob(glob) if f.is_file() and not is_binary_file(str(f))]
        else:
            return {"error": f"Path not found: {path}"}

        total_count = 0
        for f in sorted(files_to_search):
            try:
                text = f.read_text(encoding="utf-8", errors="replace")
            except (OSError, PermissionError):
                continue
            file_lines = text.splitlines()
            for i, line in enumerate(file_lines, 1):
                if regex.search(line):
                    total_count += 1
                    if total_count > offset and len(matches) < limit:
                        match_entry: Dict[str, Any] = {
                            "file": str(f), "line": i,
                            "content": _redact_content(line.rstrip()),
                        }
                        if context_lines > 0:
                            ctx_start = max(0, i - 1 - context_lines)
                            ctx_end = min(len(file_lines), i + context_lines)
                            match_entry["context"] = [
                                f"{ctx_start + j + 1}|{file_lines[ctx_start + j]}"
                                for j in range(ctx_end - ctx_start)
                            ]
                        matches.append(match_entry)

        result = {"matches": matches, "total_count": total_count, "pattern": pattern}
        if total_count > offset + limit:
            result["truncated"] = True
            result["_hint"] = f"Results truncated. Use offset={offset + limit} to see more."

    if count >= 3:
        result["_warning"] = (
            f"You have run this exact search {count} times consecutively. "
            "The results have not changed. Use the information you already have."
        )

    return result


def file_diff(path_a: str, path_b: str) -> Dict[str, Any]:
    """Compare two files and return unified diff."""
    import difflib
    try:
        a_lines = Path(path_a).expanduser().resolve().read_text(encoding="utf-8").splitlines(keepends=True)
        b_lines = Path(path_b).expanduser().resolve().read_text(encoding="utf-8").splitlines(keepends=True)
    except (OSError, PermissionError) as e:
        return {"error": str(e)}

    diff = list(difflib.unified_diff(a_lines, b_lines, fromfile=path_a, tofile=path_b))
    return {
        "diff": "".join(diff) if diff else "Files are identical.",
        "changes": len([l for l in diff if l.startswith("+") or l.startswith("-")]) - 2 if diff else 0,
    }
