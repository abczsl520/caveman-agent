"""Shared utilities for command handlers. Eliminates repeated patterns."""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from caveman.paths import CAVEMAN_HOME


def read_json(filename: str) -> dict | list | None:
    """Read a JSON file from CAVEMAN_HOME. Returns None on failure."""
    path = CAVEMAN_HOME / filename
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text())
    except Exception:
        return None


def count_files(subdir: str, pattern: str = "*.json") -> int:
    """Count files matching pattern in a CAVEMAN_HOME subdirectory."""
    d = CAVEMAN_HOME / subdir
    return len(list(d.glob(pattern))) if d.exists() else 0


def list_files(subdir: str, pattern: str = "*.json") -> list[Path]:
    """List files matching pattern in a CAVEMAN_HOME subdirectory."""
    d = CAVEMAN_HOME / subdir
    return sorted(d.glob(pattern)) if d.exists() else []


def memory_count() -> int:
    """Get total memory count from SQLite."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return 0
    try:
        conn = sqlite3.connect(str(db))
        count = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        conn.close()
        return count
    except Exception:
        return 0


def memory_search_fts(query: str, limit: int = 5) -> list[str]:
    """FTS5 search on memories. Returns list of content strings."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT content FROM memories_fts WHERE memories_fts MATCH ? LIMIT ?",
            (query, limit),
        ).fetchall()
        conn.close()
        return [r[0] for r in rows]
    except Exception:
        return []


def memory_stats() -> tuple[int, dict[str, int]]:
    """Get memory total + category breakdown."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return 0, {}
    try:
        conn = sqlite3.connect(str(db))
        total = conn.execute("SELECT COUNT(*) FROM memories").fetchone()[0]
        cats = {}
        try:
            rows = conn.execute("SELECT type, COUNT(*) FROM memories GROUP BY type").fetchall()
            cats = {r[0]: r[1] for r in rows}
        except Exception:
            pass  # intentional: non-critical
        conn.close()
        return total, cats
    except Exception:
        return 0, {}


def load_config_safe() -> dict:
    """Load config, return empty dict on failure."""
    try:
        from caveman.config.loader import load_config
        return load_config()
    except Exception:
        return {}


def memory_recent(limit: int = 5) -> list[tuple[str, str, float]]:
    """Get recent memories. Returns [(content, type, trust_score)]."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT content, type, trust_score FROM memories "
            "ORDER BY created_at DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []


def memory_top_retrieved(limit: int = 5) -> list[tuple[str, int, float]]:
    """Get most-retrieved memories. Returns [(content, retrieval_count, trust_score)]."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT content, retrieval_count, trust_score FROM memories "
            "WHERE retrieval_count > 0 ORDER BY retrieval_count DESC LIMIT ?", (limit,)
        ).fetchall()
        conn.close()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []


def memory_high_trust(limit: int = 5) -> list[tuple[str, float, str]]:
    """Get high-trust memories. Returns [(content, trust_score, created_at)]."""
    db = CAVEMAN_HOME / "memory" / "caveman.db"
    if not db.exists():
        return []
    try:
        conn = sqlite3.connect(str(db))
        rows = conn.execute(
            "SELECT content, trust_score, created_at FROM memories "
            "WHERE trust_score >= 0.7 ORDER BY trust_score DESC, created_at DESC LIMIT ?",
            (limit,)
        ).fetchall()
        conn.close()
        return [(r[0], r[1], r[2]) for r in rows]
    except Exception:
        return []
