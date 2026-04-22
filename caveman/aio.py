"""Async wrappers for blocking file I/O operations.

Thin wrappers around pathlib.Path and builtins that run in a thread pool
via asyncio.to_thread(), preventing event-loop stalls in async code.

Usage:
    from caveman.aio import aio_read_text, aio_write_text, aio_exists
    content = await aio_read_text(some_path)
"""

from __future__ import annotations

import asyncio
import os
import shutil
from pathlib import Path
from typing import Any, Iterator

__all__ = [
    "aio_read_text",
    "aio_write_text",
    "aio_read_bytes",
    "aio_write_bytes",
    "aio_exists",
    "aio_is_file",
    "aio_is_dir",
    "aio_stat",
    "aio_mkdir",
    "aio_glob",
    "aio_iterdir",
    "aio_unlink",
    "aio_rename",
    "aio_rglob",
]



# ── Read / Write ──────────────────────────────────────────────────────

async def aio_read_text(path: Path | str, encoding: str = "utf-8", errors: str | None = None) -> str:
    """Read text from a file without blocking the event loop."""
    kwargs: dict[str, str] = {"encoding": encoding}
    if errors is not None:
        kwargs["errors"] = errors
    return await asyncio.to_thread(Path(path).read_text, **kwargs)


async def aio_write_text(path: Path | str, data: str, encoding: str = "utf-8", errors: str | None = None) -> int:
    """Write text to a file without blocking the event loop."""
    kwargs: dict[str, str] = {"encoding": encoding}
    if errors is not None:
        kwargs["errors"] = errors
    return await asyncio.to_thread(Path(path).write_text, data, **kwargs)


async def aio_read_bytes(path: Path | str) -> bytes:
    """Read bytes from a file without blocking the event loop."""
    return await asyncio.to_thread(Path(path).read_bytes)


async def aio_write_bytes(path: Path | str, data: bytes) -> int:
    """Write bytes to a file without blocking the event loop."""
    return await asyncio.to_thread(Path(path).write_bytes, data)


# ── Path queries ──────────────────────────────────────────────────────

async def aio_exists(path: Path | str) -> bool:
    """Check if a path exists without blocking the event loop."""
    return await asyncio.to_thread(Path(path).exists)


async def aio_is_file(path: Path | str) -> bool:
    """Check if path is a file without blocking the event loop."""
    return await asyncio.to_thread(Path(path).is_file)


async def aio_is_dir(path: Path | str) -> bool:
    """Check if path is a directory without blocking the event loop."""
    return await asyncio.to_thread(Path(path).is_dir)


async def aio_stat(path: Path | str) -> os.stat_result:
    """Stat a path without blocking the event loop."""
    return await asyncio.to_thread(Path(path).stat)


# ── Directory operations ──────────────────────────────────────────────

async def aio_mkdir(path: Path | str, parents: bool = True, exist_ok: bool = True) -> None:
    """Create directory without blocking the event loop."""
    return await asyncio.to_thread(Path(path).mkdir, parents=parents, exist_ok=exist_ok)


async def aio_glob(path: Path | str, pattern: str) -> list[Path]:
    """Glob a directory without blocking the event loop. Returns a list (not iterator)."""
    return await asyncio.to_thread(lambda: list(Path(path).glob(pattern)))


async def aio_iterdir(path: Path | str) -> list[Path]:
    """List directory contents without blocking the event loop. Returns a list."""
    return await asyncio.to_thread(lambda: list(Path(path).iterdir()))


# ── File mutations ────────────────────────────────────────────────────

async def aio_unlink(path: Path | str, missing_ok: bool = True) -> None:
    """Delete a file without blocking the event loop."""
    return await asyncio.to_thread(Path(path).unlink, missing_ok=missing_ok)


async def aio_rename(src: Path | str, dst: Path | str) -> Path:
    """Rename/move a file without blocking the event loop."""
    return await asyncio.to_thread(Path(src).rename, dst)


# ── os.path compat ────────────────────────────────────────────────────

async def aio_rglob(path: Path | str, pattern: str) -> list[Path]:
    """Recursive glob without blocking the event loop."""
    return await asyncio.to_thread(lambda: list(Path(path).rglob(pattern)))
