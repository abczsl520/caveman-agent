"""File Operations v2 — read with pagination, unified diff.

Extracted from Hermes file_operations.py (1152 lines).
Key patterns: patch/replace with context, search with ripgrep, glob, binary detection.
"""
from __future__ import annotations

import difflib
import logging
from pathlib import Path
from typing import Any, Dict

from caveman.tools.registry import tool
from caveman.aio import aio_exists, aio_read_text, aio_stat

__all__ = [
    "MAX_FILE_SIZE",
    "MAX_SEARCH_RESULTS",
    "is_binary",
    "add_line_numbers",
    "file_read_v2",
    "file_diff",
]


logger = logging.getLogger("caveman.tools.file_ops_v2")

# Binary detection
_BINARY_EXTENSIONS = frozenset({
    ".png", ".jpg", ".jpeg", ".gif", ".webp", ".ico", ".bmp",
    ".mp3", ".mp4", ".wav", ".ogg", ".webm", ".avi",
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar",
    ".pdf", ".doc", ".docx", ".xls", ".xlsx",
    ".exe", ".dll", ".so", ".dylib", ".o",
    ".pyc", ".pyo", ".class", ".wasm",
    ".sqlite", ".db", ".sqlite3",
})

MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
MAX_SEARCH_RESULTS = 100


def is_binary(path: str) -> bool:
    """Check if a file is likely binary."""
    ext = Path(path).suffix.lower()
    if ext in _BINARY_EXTENSIONS:
        return True
    try:
        with open(path, "rb") as f:
            chunk = f.read(8192)
            return b"\x00" in chunk
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return False


def add_line_numbers(content: str, start: int = 1) -> str:
    """Add line numbers to content."""
    lines = content.split("\n")
    width = len(str(start + len(lines)))
    return "\n".join(f"{i:{width}d} | {line}" for i, line in enumerate(lines, start))


@tool(
    name="file_read_v2",
    description="Read file with line numbers, offset, and limit",
    params={
        "path": {"type": "string", "description": "File path"},
        "offset": {"type": "integer", "description": "Start line (1-based, default 1)"},
        "limit": {"type": "integer", "description": "Max lines to read (default 500)"},
        "numbered": {"type": "boolean", "description": "Add line numbers (default true)"},
    },
    required=["path"],
)
async def file_read_v2(
    path: str, offset: int = 1, limit: int = 500, numbered: bool = True,
) -> Dict[str, Any]:
    """Read file with line numbers and pagination."""
    p = Path(path).expanduser()
    if not await aio_exists(p):
        return {"ok": False, "error": f"File not found: {path}"}
    if is_binary(str(p)):
        return {"ok": False, "error": f"Binary file: {path}"}
    if (await aio_stat(p)).st_size > MAX_FILE_SIZE:
        return {"ok": False, "error": f"File too large: {(await aio_stat(p)).st_size} bytes"}

    try:
        content = await aio_read_text(p, encoding="utf-8", errors="replace")
        lines = content.split("\n")
        total = len(lines)
        start = max(1, offset) - 1
        end = min(start + limit, total)
        selected = lines[start:end]
        text = "\n".join(selected)
        if numbered:
            text = add_line_numbers(text, start + 1)
        return {
            "ok": True,
            "content": text,
            "total_lines": total,
            "showing": f"{start + 1}-{end}",
            "has_more": end < total,
        }
    except Exception as e:
        return {"ok": False, "error": str(e)}


@tool(
    name="file_diff",
    description="Show unified diff between two files or a file and a string",
    params={
        "path_a": {"type": "string", "description": "First file path"},
        "path_b": {"type": "string", "description": "Second file path (or empty for stdin)"},
        "content_b": {"type": "string", "description": "Content to compare against (if path_b empty)"},
    },
    required=["path_a"],
)
async def file_diff(
    path_a: str, path_b: str = "", content_b: str = "",
) -> Dict[str, Any]:
    """Generate unified diff."""
    try:
        a_content = await aio_read_text(Path(path_a).expanduser(), encoding="utf-8")
        if path_b:
            b_content = await aio_read_text(Path(path_b).expanduser(), encoding="utf-8")
        elif content_b:
            b_content = content_b
        else:
            return {"ok": False, "error": "Need path_b or content_b"}

        diff = list(difflib.unified_diff(
            a_content.splitlines(keepends=True),
            b_content.splitlines(keepends=True),
            fromfile=path_a,
            tofile=path_b or "(input)",
        ))
        return {"ok": True, "diff": "".join(diff), "changed": len(diff) > 0}
    except Exception as e:
        return {"ok": False, "error": str(e)}
