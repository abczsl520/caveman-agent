"""Obsidian Vault Bridge — live sync between Obsidian and Caveman's memory.

See obsidian_models.py for data models (ObsidianNote, ObsidianVault,
VaultSyncState, SyncResult) and parsing utilities.
"""
from __future__ import annotations

import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from caveman.aio import aio_exists, aio_mkdir, aio_write_text
from caveman.import_.obsidian_models import (
    ObsidianVault, VaultSyncState, SyncResult, _split_note_for_memory,
)

logger = logging.getLogger(__name__)

class ObsidianBridge:
    """Bridges an Obsidian vault with Caveman's memory store.

    Supports:
    - Link: register a vault path
    - Sync: incremental sync (only changed files)
    - Full sync: re-index everything
    - Write-back: Caveman writes a note to the vault
    - Status: show sync state
    """

    def __init__(self, caveman_home: Path):
        self.caveman_home = caveman_home
        self.state = VaultSyncState.load(caveman_home)

    @property
    def is_linked(self) -> bool:
        return bool(self.state.vault_path) and Path(self.state.vault_path).is_dir()

    @property
    def vault_path(self) -> Path | None:
        if self.state.vault_path:
            return Path(self.state.vault_path)
        return None

    def link(self, vault_path: Path) -> str:
        """Link a vault to Caveman."""
        vault_path = vault_path.expanduser().resolve()
        if not vault_path.is_dir():
            return f"Error: {vault_path} is not a directory"
        # Verify it looks like an Obsidian vault
        obsidian_dir = vault_path / ".obsidian"
        if not obsidian_dir.is_dir():
            return f"Warning: {vault_path} has no .obsidian/ dir — linking anyway"
        self.state.vault_path = str(vault_path)
        self.state.save(self.caveman_home)
        return f"Linked vault: {vault_path}"

    def unlink(self) -> str:
        """Unlink the vault."""
        self.state = VaultSyncState()
        self.state.save(self.caveman_home)
        return "Vault unlinked"

    async def sync(
        self,
        memory_manager: Any,
        full: bool = False,
        dry_run: bool = False,
    ) -> SyncResult:
        """Sync vault changes to memory store.

        Args:
            memory_manager: Caveman MemoryManager
            full: If True, re-index everything (ignore sync state)
            dry_run: If True, preview only
        """
        result = SyncResult()

        if not self.is_linked:
            result.errors.append("No vault linked. Use `caveman vault link <path>`")
            return result

        vault = ObsidianVault(root=Path(self.state.vault_path))
        vault.scan()

        if full:
            # Reset sync state for full re-index
            old_state = self.state.files.copy()
            self.state.files.clear()
        else:
            old_state = {}

        current_paths = set(vault.notes.keys())

        # --- Handle deleted files ---
        deleted_records = self.state.get_deleted(current_paths)
        for rec in deleted_records:
            if not dry_run:
                # Remove memories associated with deleted file
                for mid in rec.memory_ids:
                    try:
                        await memory_manager.delete(mid)
                    except Exception:
                        pass  # Best effort
                del self.state.files[rec.relative_path]
            result.deleted += 1

        # --- Handle new and changed files ---
        for rel_path, note in vault.notes.items():
            if not self.state.needs_sync(note) and not full:
                result.unchanged += 1
                continue

            is_update = rel_path in self.state.files
            chunks = _split_note_for_memory(note)
            if not chunks:
                result.unchanged += 1
                continue

            memory_ids: list[str] = []
            try:
                # If updating, delete old memories first
                if is_update and not dry_run:
                    old_rec = self.state.files.get(rel_path)
                    if old_rec:
                        for mid in old_rec.memory_ids:
                            try:
                                await memory_manager.delete(mid)
                            except Exception:
                                pass  # intentional: Exception suppressed

                # Store new chunks
                for chunk in chunks:
                    if not dry_run:
                        mid = await memory_manager.store(
                            chunk,
                            note.inferred_memory_type,
                            metadata={
                                "source": "obsidian",
                                "vault_file": rel_path,
                                "title": note.title,
                                "tags": note.tags,
                                "date": note.date,
                                "folder": note.folder,
                                "synced_at": datetime.now().isoformat(),
                            },
                        )
                        memory_ids.append(mid)
                    else:
                        memory_ids.append("dry-run")

                if not dry_run:
                    self.state.mark_synced(note, memory_ids)

                if is_update:
                    result.updated += 1
                else:
                    result.added += 1

            except Exception as e:
                result.failed += 1
                result.errors.append(f"{rel_path}: {e}")
                logger.warning("Sync failed for %s: %s", rel_path, e)

        if not dry_run:
            self.state.save(self.caveman_home)

        return result

    async def write_note(
        self,
        title: str,
        content: str,
        folder: str = "",
        tags: list[str] | None = None,
    ) -> Path | None:
        """Write a note back to the Obsidian vault.

        Used when Caveman learns something worth persisting as a human-readable note.
        """
        if not self.is_linked:
            return None

        vault_root = Path(self.state.vault_path)

        # Build front matter
        fm: dict[str, Any] = {}
        if tags:
            fm["tags"] = tags
        fm["date"] = datetime.now().strftime("%Y-%m-%d")
        fm["source"] = "caveman"

        # Build file content
        fm_str = yaml.dump(fm, allow_unicode=True, default_flow_style=False).strip()
        full_content = f"---\n{fm_str}\n---\n\n{content}"

        # Determine path
        safe_title = re.sub(r'[<>:"/\\|?*]', "", title)
        if folder:
            target_dir = vault_root / folder
        else:
            target_dir = vault_root
        await aio_mkdir(target_dir, parents=True, exist_ok=True)
        target = target_dir / f"{safe_title}.md"

        # Don't overwrite existing notes
        if await aio_exists(target):
            # Append timestamp to make unique
            ts = datetime.now().strftime("%H%M%S")
            target = target_dir / f"{safe_title}-{ts}.md"

        await aio_write_text(target, full_content, encoding="utf-8")
        return target

    def status(self) -> dict[str, Any]:
        """Get current sync status."""
        if not self.is_linked:
            return {"linked": False}

        vault = ObsidianVault(root=Path(self.state.vault_path))
        vault.scan()

        current_paths = set(vault.notes.keys())
        synced_paths = set(self.state.files.keys())

        new_files = current_paths - synced_paths
        deleted_files = synced_paths - current_paths
        changed_files = {
            p for p in current_paths & synced_paths
            if self.state.needs_sync(vault.notes[p])
        }

        return {
            "linked": True,
            "vault_path": self.state.vault_path,
            "last_sync": self.state.last_sync,
            "total_notes": vault.note_count,
            "total_size_kb": round(vault.total_size / 1024, 1),
            "synced_files": len(synced_paths & current_paths),
            "new_files": len(new_files),
            "changed_files": len(changed_files),
            "deleted_files": len(deleted_files),
            "pending_changes": len(new_files) + len(changed_files) + len(deleted_files),
        }
