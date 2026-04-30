"""Operator-facing memory source governance CLI."""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer

from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.paths import MEMORY_DB_PATH
from caveman.training._flywheel_memory_diagnostics import _collect_memory_source_policy_drift

app = typer.Typer(help="Preview memory source policy candidates.")


def _store(db: Optional[Path]) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=db or MEMORY_DB_PATH)


@app.command("preview-drift")
def preview_drift(
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite memory DB path."),
    min_rows: int = typer.Option(3, "--min-rows", min=1, help="Minimum rows before flagging a source."),
    limit: int = typer.Option(8, "--limit", min=1, max=100, help="Maximum candidates to show."),
) -> None:
    """Preview unmanaged low-signal import sources without mutating memory rows."""
    store = _store(db)
    try:
        conn = store._get_conn()  # noqa: SLF001 - CLI needs dashboard diagnostic connection.
        candidates = _collect_memory_source_policy_drift(conn, min_rows=min_rows, limit=limit)
    finally:
        store.close()

    if not candidates:
        typer.echo("No unmanaged low-signal import source drift candidates found.")
        return

    typer.echo(f"candidate_count={len(candidates)}")
    for idx, row in enumerate(candidates, 1):
        typer.echo(f"{idx}. source={row['label']} total={row['total']} active={row['active']}")
        typer.echo(
            "   "
            f"avg_trust={row['avg_trust']} never_recalled_pct={row['never_recalled_pct']} "
            f"helpful_pct={row['helpful_pct']}"
        )
        typer.echo(f"   reason={row['reason']} recommended_action={row['recommended_action']}")
        typer.echo(f"   candidate_policy_entry={row['candidate_policy_entry']}")
