"""Operator CLI for reversible memory quarantine lifecycle."""
from __future__ import annotations

import asyncio
from pathlib import Path
from typing import Optional

import typer

from caveman.memory.sqlite_store import SQLiteMemoryStore
from caveman.memory.quarantine import list_quarantined as quarantine_list
from caveman.memory.quarantine import preview_restore_quarantined as quarantine_preview
from caveman.memory.quarantine import restore_quarantined as quarantine_restore
from caveman.paths import MEMORY_DB_PATH

app = typer.Typer(help="Review and restore quarantined memories.")


def _store(db: Optional[Path]) -> SQLiteMemoryStore:
    return SQLiteMemoryStore(db_path=db or MEMORY_DB_PATH)


@app.command(name="list")
def list_quarantined(
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite memory DB path"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by metadata source"),
    limit: int = typer.Option(50, "--limit", min=1, max=500, help="Maximum rows to show"),
) -> None:
    """List quarantined memories with audit clues."""
    store = _store(db)
    try:
        rows = quarantine_list(store, source=source, limit=limit)
        if not rows:
            typer.echo("No quarantined memories found.")
            return
        for entry in rows:
            meta = entry.metadata
            reason = meta.get("quarantine_reason", "")
            source_label = meta.get("source", "")
            typer.echo(f"{entry.id}	{source_label}	{reason}	{entry.content[:120]}")
    finally:
        store.close()


@app.command(name="preview-restore")
def preview_restore(
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite memory DB path"),
    source: Optional[str] = typer.Option(None, "--source", help="Filter by metadata source"),
    reason: Optional[str] = typer.Option(None, "--reason", help="Filter by quarantine reason"),
    limit: int = typer.Option(500, "--limit", min=1, max=5000, help="Maximum rows to preview"),
) -> None:
    """Dry-run a scoped restore and report impact without mutating rows."""
    store = _store(db)
    try:
        preview = quarantine_preview(store, source=source, reason=reason, limit=limit)
        typer.echo(f"would_restore={preview.total_matches}")
        if preview.by_source:
            typer.echo(
                "sources "
                + " ".join(f"{key}={value}" for key, value in sorted(preview.by_source.items()))
            )
        if preview.by_reason:
            typer.echo(
                "reasons "
                + " ".join(f"{key}={value}" for key, value in sorted(preview.by_reason.items()))
            )
        for entry in preview.entries:
            meta = entry.metadata
            typer.echo(
                f"{entry.id}\t{meta.get('source', '')}\t"
                f"{meta.get('quarantine_reason', '')}\t{entry.content[:120]}"
            )
    finally:
        store.close()


@app.command()
def restore(
    memory_id: str = typer.Argument(..., help="Memory id to restore"),
    db: Optional[Path] = typer.Option(None, "--db", help="SQLite memory DB path"),
    restored_by: str = typer.Option("operator", "--by", help="Actor restoring the memory"),
    reason: str = typer.Option("manual restore", "--reason", help="Audit reason for restore"),
) -> None:
    """Restore one quarantined memory to active recall with audit metadata."""
    store = _store(db)
    try:
        restored = asyncio.run(
            quarantine_restore(
                store,
                memory_id,
                restored_by=restored_by,
                restore_reason=reason,
            )
        )
        if not restored:
            raise typer.Exit(code=1)
        typer.echo(f"restored {memory_id}")
    finally:
        store.close()
