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

from caveman.result import Ok, Err
from caveman.tools.registry import tool
import logging
from caveman.aio import aio_exists, aio_glob, aio_is_dir, aio_is_file, aio_mkdir, aio_read_text, aio_stat

__all__ = [
    "MAX_READ_LINES",
    "MAX_SEARCH_RESULTS",
    "file_read",
    "file_write",
    "file_edit",
    "file_search",
    "file_list",
]


logger = logging.getLogger(__name__)

# Paths that should never be written to
_WRITE_DENIED = {
    "/etc/passwd", "/etc/shadow", "/etc/sudoers",
    "/etc/hosts", "/etc/resolv.conf",
    "/etc/crontab",
    "/etc/cron.d/",
    "~/.ssh/",
    "~/.bashrc",
    "~/.bash_profile",
    "~/.zshrc",
    "~/.profile",
    "~/.config/systemd/",
}
_WRITE_DENIED_PREFIXES = ("/proc/", "/sys/", "/dev/", "/boot/")

# Binary file extensions
_BINARY_EXTENSIONS = frozenset({
    # Images
    ".png", ".jpg", ".jpeg", ".gif", ".bmp", ".ico", ".webp", ".svg", ".tiff", ".tif",
    # Videos
    ".mp4", ".mov", ".avi", ".mkv", ".webm", ".wmv", ".flv", ".m4v", ".mpeg", ".mpg",
    # Audio
    ".mp3", ".wav", ".ogg", ".flac", ".aac", ".m4a", ".wma", ".aiff", ".opus",
    # Archives
    ".zip", ".tar", ".gz", ".bz2", ".7z", ".rar", ".xz", ".z", ".tgz", ".iso",
    # Executables/binaries
    ".exe", ".dll", ".so", ".dylib", ".bin", ".o", ".a", ".obj", ".lib",
    ".app", ".msi", ".deb", ".rpm",
    # Documents
    ".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".odt", ".ods", ".odp",
    # Fonts
    ".ttf", ".otf", ".woff", ".woff2", ".eot",
    # Bytecode / VM artifacts
    ".pyc", ".pyo", ".class", ".jar", ".war", ".ear", ".node", ".wasm", ".rlib",
    # Database files
    ".sqlite", ".sqlite3", ".db", ".mdb", ".idx",
    # Design / 3D
    ".psd", ".ai", ".eps", ".sketch", ".fig", ".xd", ".blend",
    # PDF (binary format, even though text-extractable)
    ".pdf",
    # Lock/profiling data
    ".lockb", ".dat", ".data",
})

MAX_READ_LINES = 2000
MAX_SEARCH_RESULTS = 50


def _is_write_denied(path: str) -> bool:
    """Check if path is in the write-denied list."""
    resolved = str(Path(path).expanduser().resolve())
    # Check exact matches and directory prefixes
    for denied in _WRITE_DENIED:
        denied_resolved = str(Path(denied).expanduser().resolve())
        if denied.endswith("/"):
            # Directory prefix — block anything under it
            if resolved.startswith(denied_resolved):
                return True
        elif resolved == denied_resolved:
            return True
    return any(resolved.startswith(str(Path(p).expanduser().resolve())) for p in _WRITE_DENIED_PREFIXES)


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
    description="Read a text file with line numbers and pagination. "
    "Detects binary files, blocks device paths, deduplicates unchanged re-reads, "
    "and warns on consecutive read loops.",
    params={
        "path": {"type": "string", "description": "File path to read"},
        "offset": {"type": "integer", "description": "Start line (1-based, default 1)"},
        "limit": {"type": "integer", "description": "Max lines to read (default 500, max 2000)"},
    },
    required=["path"],
)
async def file_read(
    path: str = "", offset: int = 1, limit: int = 500, **kwargs,
) -> dict[str, Any]:
    """Read file with full guards: device-path blocking, binary detection,
    read dedup, consecutive-loop detection (warn@3, block@4), staleness
    tracking, file-not-found suggestions, and secret redaction.

    Delegates to file_tools.read_file for the complete implementation.
    """
    # Accept file_path alias — LLMs often use it
    path = path or kwargs.get("file_path", "")
    if not path:
        return Err("Missing path (the file to read)")
    if offset < 1:
        return Err("offset must be >= 1")
    if limit < 1 or limit > MAX_READ_LINES:
        return Err(f"limit must be between 1 and {MAX_READ_LINES}")
    from caveman.tools.builtin.file_tools import read_file
    result = read_file(path, offset, limit, task_id=kwargs.get("task_id", "default"))
    if "error" in result:
        return Err(result["error"])
    # Re-format content with padded line numbers and box-drawing separator
    # file_tools returns "N|line" format; we upgrade to "  N │ line" for readability
    raw = result.get("content", "")
    reformatted_lines = []
    for raw_line in raw.splitlines():
        pipe_idx = raw_line.find("|")
        if pipe_idx >= 0:
            num_part = raw_line[:pipe_idx]
            text_part = raw_line[pipe_idx + 1:]
            reformatted_lines.append(f"{num_part:>4} │ {text_part}")
        else:
            reformatted_lines.append(raw_line)
    result["content"] = "\n".join(reformatted_lines)
    # Backward-compat: add integer "lines" count from "lines_shown" range
    if "lines_shown" in result and "lines" not in result:
        try:
            lo, hi = result["lines_shown"].split("-")
            result["lines"] = int(hi) - int(lo) + 1
        except (ValueError, AttributeError):
            pass
    # Wrap as ToolResult for backward compatibility
    return Ok(result)


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
        await aio_mkdir(p.parent, parents=True, exist_ok=True)
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
    path = path if path is not None else kwargs.get("file_path", "")
    old_str = old_str if old_str is not None else kwargs.get("old_string", "")
    new_str = new_str if new_str is not None else kwargs.get("new_string", "")
    if not path:
        return Err("Missing path (the file to edit)")
    if not old_str:
        return Err("Missing old_str (the text to find and replace)")
    if _is_write_denied(path):
        return Err(f"⛔ Write denied: {path}")

    p = Path(path).expanduser()
    if not await aio_exists(p):
        return Err(f"File not found: {path}")

    try:
        text = await aio_read_text(p, encoding="utf-8")
        count = text.count(old_str)
        if count == 0:
            # Fuzzy match fallback — try 8 strategies before giving up
            try:
                from caveman.tools.fuzzy_match import fuzzy_find_and_replace
                new_text, fcount, strategy, ferr = fuzzy_find_and_replace(text, old_str, new_str)
                if fcount > 0 and not ferr:
                    # Atomic write
                    tmp = p.with_suffix(p.suffix + ".tmp")
                    with open(tmp, "w", encoding="utf-8") as f:
                        f.write(new_text)
                        f.flush()
                        os.fsync(f.fileno())
                    tmp.replace(p)
                    return Ok(path=str(p.resolve()), note=f"fuzzy match ({strategy})")
            except Exception as exc:
                logger.debug("file_edit: suppressed %s", exc)

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
    description="Search file contents (regex) or find files by name (glob). "
    "Detects consecutive search loops. Supports context lines and pagination.",
    params={
        "pattern": {"type": "string", "description": "Regex for content search, or glob for file search"},
        "path": {"type": "string", "description": "Directory or file to search (default '.')"},
        "target": {"type": "string", "description": "'content' or 'files' (default 'content')"},
        "include": {"type": "string", "description": "File glob filter (e.g. '*.py')"},
        "limit": {"type": "integer", "description": "Max results (default 50)"},
        "offset": {"type": "integer", "description": "Skip first N results (default 0)"},
        "context": {"type": "integer", "description": "Context lines around matches (default 0)"},
    },
    required=["pattern"],
)
async def file_search(
    pattern: str, path: str = ".", target: str = "content",
    include: str | None = None, limit: int = 50, offset: int = 0,
    context: int = 0, **kwargs,
) -> dict[str, Any]:
    """Search files with loop detection, pagination, and context lines.

    Delegates to file_tools_search.search_in_file for the complete implementation.
    """
    if not pattern:
        return {"error": "pattern is required"}
    # ReDoS protection: reject overly complex patterns
    if len(pattern) > 200:
        return {"error": "Pattern too long (max 200 chars)"}
    if re.search(r'\([^)]*[+*][^)]*\)[+*]', pattern):
        return {"error": "Pattern contains nested quantifiers (potential ReDoS)"}
    from caveman.tools.builtin.file_tools import search_in_file
    result = search_in_file(
        pattern, path, target=target, file_glob=include,
        limit=limit, offset=offset, context_lines=context,
        task_id=kwargs.get("task_id", "default"),
    )
    if "error" in result:
        return result
    # Backward-compat: "total_count" → "count", add "files_searched"/"truncated"
    if "total_count" in result and "count" not in result:
        result["count"] = result["total_count"]
    if "truncated" not in result:
        result["truncated"] = result.get("count", 0) >= limit
    if "files_searched" not in result:
        seen = {m.get("file") for m in result.get("matches", [])}
        result["files_searched"] = len(seen)
    return result


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
    if not await aio_exists(p):
        return {"error": f"Path not found: {path}"}

    try:
        glob_fn = p.rglob if recursive else p.glob
        entries = []
        for f in sorted(glob_fn(pattern)):
            try:
                stat = await aio_stat(f)
                entries.append({
                    "path": str(f),
                    "type": "dir" if await aio_is_dir(f) else "file",
                    "size": stat.st_size,
                })
            except (PermissionError, OSError):
                continue

        return {"entries": entries, "count": len(entries)}
    except Exception as e:
        return {"error": str(e)}
