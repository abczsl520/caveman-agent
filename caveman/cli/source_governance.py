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
        all_candidates = _collect_memory_source_policy_drift(conn, min_rows=min_rows, limit=None)
    finally:
        store.close()

    if not all_candidates:
        typer.echo("No unmanaged low-signal import source drift candidates found.")
        return

    candidates = all_candidates[:limit]
    typer.echo(f"candidate_count={len(all_candidates)}")
    typer.echo(f"showing_count={len(candidates)}")
    for idx, row in enumerate(candidates, 1):
        typer.echo(f"{idx}. source={row['label']} total={row['total']} active={row['active']}")
        typer.echo(
            "   "
            f"avg_trust={row['avg_trust']} never_recalled_pct={row['never_recalled_pct']} "
            f"helpful_pct={row['helpful_pct']}"
        )
        typer.echo(f"   reason={row['reason']} recommended_action={row['recommended_action']}")
        typer.echo(f"   candidate_policy_entry={row['candidate_policy_entry']}")

    typer.echo("Policy workflow (copy/paste):")
    typer.echo("1. Review source quality outside the CLI; this command is read-only.")
    typer.echo("2. If approved, add to caveman.memory.sources.SOURCE_POLICY_LOW_SIGNAL_IMPORTS:")
    for row in candidates:
        typer.echo(f"   {row['candidate_policy_entry']!r},")
    typer.echo(f"3. Re-run: caveman source-governance preview-drift --min-rows {min_rows}")
    typer.echo("auto_mutation=disabled")
