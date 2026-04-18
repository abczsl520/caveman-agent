"""Memory importer — orchestrator for the modular import system.

`caveman import --from openclaw` reads data from external sources
and writes them into Caveman's Memory Store.

Delegates to caveman.import_.* modules for source-specific logic.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from caveman.import_ import IMPORTERS
from caveman.import_.base import ImportManifest, ImportResult
from caveman.import_.report import (
    format_manifest_report, format_result_report, format_detect_report,
)
from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)


def detect_sources() -> dict[str, bool]:
    """Detect which import sources are available on this system."""
    results: dict[str, bool] = {}
    for name, cls in IMPORTERS.items():
        if name == "directory":
            continue
        try:
            importer = cls(caveman_home=CAVEMAN_HOME)
            results[name] = importer.detect()
        except Exception:
            results[name] = False
    return results


async def import_memories(
    source: str,
    memory_manager: Any,
    directory: str | None = None,
    dry_run: bool = True,
    only: str | None = None,
    include_secrets: bool = False,
) -> ImportResult:
    """Import memories from an external source.

    Args:
        source: Source name (openclaw, hermes, codex, claude-code, directory)
        memory_manager: Target MemoryManager
        directory: Custom directory path (for source="directory")
        dry_run: If True, preview only (default)
        only: Filter to "memory", "config", or "workspace" only
        include_secrets: If True, import entries containing secrets too
    """
    if source not in IMPORTERS:
        result = ImportResult()
        result.details.append(
            f"Unknown source: {source}. Available: {', '.join(IMPORTERS.keys())}"
        )
        return result

    cls = IMPORTERS[source]
    if source == "directory":
        from caveman.import_.directory import DirectoryImporter
        importer = DirectoryImporter(
            caveman_home=CAVEMAN_HOME, dry_run=dry_run,
            directory=Path(directory).expanduser() if directory else None,
            include_secrets=include_secrets,
        )
    else:
        importer = cls(caveman_home=CAVEMAN_HOME, dry_run=dry_run, include_secrets=include_secrets)

    if not importer.detect():
        result = ImportResult()
        result.details.append(f"Source not found: {source}")
        return result

    manifest = importer.scan()

    # Filter by --only
    if only:
        manifest.items = [i for i in manifest.items if i.target_type == only]

    return await importer.execute(manifest, memory_manager)
