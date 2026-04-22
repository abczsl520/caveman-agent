"""Centralized SQLite connection factory.

Every sqlite3.connect() in the codebase should go through this module.
Ensures consistent PRAGMA settings (WAL, busy_timeout, foreign_keys)
and provides a single place to tune SQLite behavior.

Usage:
    from caveman.db import connect
    conn = connect("path/to/db.sqlite")
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

# Defaults — tuned for concurrent async access
from caveman.timeouts import SQLITE_BUSY, SQLITE_CONNECT

BUSY_TIMEOUT_MS = int(SQLITE_BUSY)  # From timeouts.py, env-configurable
JOURNAL_MODE = "WAL"     # Write-Ahead Logging for concurrent reads


def connect(
    path: str | Path,
    *,
    busy_timeout: int = BUSY_TIMEOUT_MS,
    journal_mode: str = JOURNAL_MODE,
    foreign_keys: bool = True,
    row_factory: type | None = None,
    isolation_level: str | None = "DEFERRED",
    **kwargs,
) -> sqlite3.Connection:
    """Create a SQLite connection with production-safe defaults.

    All PRAGMA settings are applied automatically. Callers don't need
    to remember WAL/busy_timeout/foreign_keys.
    """
    conn = sqlite3.connect(str(path), timeout=SQLITE_CONNECT, isolation_level=isolation_level, **kwargs)
    conn.execute(f"PRAGMA journal_mode={journal_mode}")
    conn.execute(f"PRAGMA busy_timeout={busy_timeout}")
    if foreign_keys:
        conn.execute("PRAGMA foreign_keys=ON")
    if row_factory:
        conn.row_factory = row_factory
    return conn
