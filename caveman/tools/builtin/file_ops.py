"""File operation tools — read, write, edit, search, list.

Production-grade file operations with line numbers, binary detection,
search (grep-like), and write-denied path protection.

Ported patterns from Hermes file_operations.py (MIT, Nous Research).
"""
from __future__ import annotations

import os
import re
from pathlib import Path
from typing import Any

from caveman.result import ToolResult, Ok, Err
from caveman.tools.registry import tool

# Paths that should never be written to
_WRITE_DENIED = {
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/hosts", "/etc/resolv.conf",
}
_WRITE_DENIED_PREFIXES = ("/proc/", "/sys/", "/dev/")

# Binary file extensions
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg",
    ".mp3", ".mp4", ".wav", ".avi", ".mov", ".mkv",
    ".zip", ".tar", ".gz", ".bz2", ".xz", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx",
    ".exe", ".dll", ".so", ".dylib", ".o", ".a",
    ".pyc", ".pyo", ".class", ".wasm",
    ".sqlite", ".db", ".sqlite3",
})

MAX_READ_LINES = 2000
MAX_SEARCH_RESULTS = 50


def _is_write_denied(path: str) -> bool:
    """Check if path is in the write-denied list."""
    resolved = str(Path(path).expanduser().resolve())
    # Check exact matches (resolve symlinks like /etc → /private/etc on macOS)
    for denied in _WRITE_DENIED:
        denied_resolved = str(Path(denied).resolve())
        if resolved == denied_resolved:
            return True
    return any(resolved.startswith(str(Path(p).resolve())) for p in _WRITE_DENIED_PREFIXES)


def _is_binary(path: Path) -> bool:
    """Check if file is likely binary."""
    return path.suffix.lower() in _BINARY_EXTENSIONS


def _add_line_numbers(content: str, start: int = 1) -> str:
    """Add line numbers to content."""
    lines = content.splitlines()
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:{width}d} │ {line}" for i, line in enumerate(lines, start))


@tool(
    name="file_read",
    description="Read a file with line numbers. Supports offset and limit.",
    params={
        "path": {"type": "string", "description": "File path to read"},
        "offset": {"type": "integer", "description": "Start line (1-based, default 1)"},
        "limit": {"type": "integer", "description": "Max lines to read (default 500)"},
    },
    required=["path"],
)
async def file_read(
    path: str = "", offset: int = 1, limit: int = 500, **kwargs,
) -> dict[str, Any]:
    """Read file with line numbers, offset, and limit."""
    # Accept file_path alias — LLMs often use it
    path = path or kwargs.get("file_path", "")
    if not path:
        return Err("Missing path (the file to read)")
    p = Path(path).expanduser()
    if not p.exists():
        return Err(f"File not found: {path}")
    if _is_binary(p):
        size = p.stat().st_size
        return Err(f"Binary file ({p.suffix}, {size:,} bytes). Use bash to inspect.")

    try:
        text = p.read_text(encoding="utf-8", errors="replace")
        all_lines = text.splitlines()
        total = len(all_lines)

        # 1-based offset
        start_idx = max(0, offset - 1)
        end_idx = start_idx + min(limit, MAX_READ_LINES)
        selected = all_lines[start_idx:end_idx]

        numbered = _add_line_numbers("\n".join(selected), start=start_idx + 1)
        truncated = end_idx < total

        return Ok(
            content=numbered,
            lines=len(selected),
            total_lines=total,
            path=str(p.resolve()),
            truncated=truncated,
        )
    except Exception as e:
        return Err(str(e))


@tool(
    name="file_write",
    description="Write content to a file. Creates parent directories.",
    params={
        "path": {"type": "string", "description": "File path to write"},
        "content": {"type": "string", "description": "Content to write"},
    },
    required=["path", "content"],
)
async def file_write(path: str, content: str) -> dict[str, Any]:
    """Write content to file with safety checks."""
    if _is_write_denied(path):
        return Err(f"⛔ Write denied: {path}")

    p = Path(path).expanduser().resolve()
    try:
        p.parent.mkdir(parents=True, exist_ok=True)
        # Atomic write: tmp → fsync → rename
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(content)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)  # atomic on POSIX
        lines = content.count("\n") + (1 if content and not content.endswith("\n") else 0)
        return Ok(
            path=str(p.resolve()),
            bytes=len(content.encode("utf-8")),
            lines=lines,
        )
    except Exception as e:
        return Err(str(e))


@tool(
    name="file_edit",
    description="Replace exact text in a file. Fails if old_str not found or matches multiple times.",
    params={
        "path": {"type": "string", "description": "File path to edit"},
        "old_str": {"type": "string", "description": "Exact text to find"},
        "new_str": {"type": "string", "description": "Replacement text"},
    },
    required=["path", "old_str", "new_str"],
)
async def file_edit(path: str = "", old_str: str = "", new_str: str = "", **kwargs) -> dict[str, Any]:
    """Replace exact string in file. Single occurrence only for safety."""
    # Accept common aliases — LLMs often use file_path, old_string, new_string
    path = path or kwargs.get("file_path", "")
    old_str = old_str or kwargs.get("old_string", "")
    new_str = new_str or kwargs.get("new_string", "")
    if not path:
        return Err("Missing path (the file to edit)")
    if not old_str:
        return Err("Missing old_str (the text to find and replace)")
    if _is_write_denied(path):
        return Err(f"⛔ Write denied: {path}")

    p = Path(path).expanduser()
    if not p.exists():
        return Err(f"File not found: {path}")

    try:
        text = p.read_text(encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            # Help debug: show similar lines
            lines = text.splitlines()
            old_lines = old_str.splitlines()
            old_first = old_lines[0].strip() if old_lines else ""
            # Match on first significant word(s) for fuzzy similarity
            words = old_first.split()[:2]
            prefix = " ".join(words) if words else ""
            similar = [
                f"  L{i+1}: {l.strip()[:80]}"
                for i, l in enumerate(lines)
                if prefix and prefix in l
            ][:3]
            hint = "\nSimilar lines:\n" + "\n".join(similar) if similar else ""
            return Err(f"old_str not found in file.{hint}")
        if count > 1:
            return Err(f"old_str found {count} times. Be more specific to avoid ambiguity.")

        new_text = text.replace(old_str, new_str, 1)
        # Atomic write: tmp → fsync → rename
        tmp = p.with_suffix(p.suffix + ".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(new_text)
            f.flush()
            os.fsync(f.fileno())
        tmp.replace(p)
        return Ok(path=str(p.resolve()))
    except Exception as e:
        return Err(str(e))


@tool(
    name="file_search",
    description="Search for pattern in files (like grep). Supports regex.",
    params={
        "pattern": {"type": "string", "description": "Search pattern (regex supported)"},
        "path": {"type": "string", "description": "Directory or file to search (default '.')"},
        "include": {"type": "string", "description": "File glob pattern (e.g. '*.py')"},
    },
    required=["pattern"],
)
async def file_search(
    pattern: str, path: str = ".", include: str | None = None,
) -> dict[str, Any]:
    """Search for pattern in files. Returns matching lines with context."""
    root = Path(path).expanduser()
    if not root.exists():
        return {"error": f"Path not found: {path}"}

    try:
        regex = re.compile(pattern, re.IGNORECASE)
    except re.error as e:
        return {"error": f"Invalid regex: {e}"}

    matches: list[dict] = []
    files_searched = 0

    if root.is_file():
        files = [root]
    else:
        glob_pattern = include or "**/*"
        files = sorted(root.glob(glob_pattern))

    for f in files:
        if not f.is_file() or _is_binary(f):
            continue
        files_searched += 1
        try:
            text = f.read_text(encoding="utf-8", errors="replace")
            for i, line in enumerate(text.splitlines(), 1):
                if regex.search(line):
                    matches.append({
                        "file": str(f),
                        "line": i,
                        "text": line.strip()[:200],
                    })
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        return {
                            "matches": matches,
                            "count": len(matches),
                            "files_searched": files_searched,
                            "truncated": True,
                        }
        except (UnicodeDecodeError, PermissionError):
            continue

    return {
        "matches": matches,
        "count": len(matches),
        "files_searched": files_searched,
        "truncated": False,
    }


@tool(
    name="file_list",
    description="List files in a directory with sizes.",
    params={
        "path": {"type": "string", "description": "Directory path (default '.')"},
        "pattern": {"type": "string", "description": "Glob pattern (default '*')"},
        "recursive": {"type": "boolean", "description": "Recurse into subdirectories"},
    },
)
async def file_list(
    path: str = ".", pattern: str = "*", recursive: bool = False, **kwargs,
) -> dict[str, Any]:
    """List files with sizes and types."""
    # Accept directory alias — LLMs often use it
    path = path if path != "." else kwargs.get("directory", path)
    p = Path(path).expanduser()
    if not p.exists():
        return {"error": f"Path not found: {path}"}

    try:
        glob_fn = p.rglob if recursive else p.glob
        entries = []
        for f in sorted(glob_fn(pattern)):
            try:
                stat = f.stat()
                entries.append({
                    "path": str(f),
                    "type": "dir" if f.is_dir() else "file",
                    "size": stat.st_size,
                })
            except (PermissionError, OSError):
                continue

        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": str(e)}
