"""Migration CLI helpers."""
from __future__ import annotations

import sqlite3
from pathlib import Path

from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.store_helpers import get_schema_version, migrate_schema, pending_migrations
from caveman.paths import MEMORY_DB_PATH


__all__ = ["run_migrate"]


def run_migrate(db_path: str | Path | None = None, dry_run: bool = True) -> str:
    """Run or preview memory database migrations."""
    path = Path(db_path).expanduser() if db_path else MEMORY_DB_PATH
    path.parent.mkdir(parents=True, exist_ok=True)

    if not dry_run:
        # Ensure the baseline SQLite schema exists before applying additive
        # migrations. A brand-new DB should migrate cleanly, but a legacy DB
        # must still flow through explicit numbered migrations.
        bootstrap_conn = sqlite3.connect(path)
        try:
            has_memories = bootstrap_conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='memories'"
            ).fetchone() is not None
        finally:
            bootstrap_conn.close()
        if not has_memories:
            store = SQLiteMemoryStore(path)
            store._get_conn()
            store.close()

    conn = sqlite3.connect(path)
    try:
        current = get_schema_version(conn)
        pending = pending_migrations(conn)
        if dry_run:
            if not pending:
                return f"Memory schema v{current}: no pending migrations."
            lines = [f"Memory schema v{current}: {len(pending)} pending migration(s):"]
            lines.extend(f"  - v{version}: {name}" for version, name in pending)
            return "\n".join(lines)

        applied = migrate_schema(conn, dry_run=False)
        final = get_schema_version(conn)
        if not applied:
            return f"Memory schema v{final}: already up to date."
        lines = [f"Applied {len(applied)} migration(s). Memory schema is now v{final}:"]
        lines.extend(f"  - v{version}: {name}" for version, name in applied)
        return "\n".join(lines)
    finally:
        conn.close()
