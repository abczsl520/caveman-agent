"""Workspace → Memory Store sync — keeps vector DB in sync with workspace files.

PRD §8.8.1: MEMORY.md should be "读取 + 导入到 Memory Store".
This module makes it automatic: on session start, detect changes in
workspace files and incrementally sync them to the vector database.

Design principles:
  - Incremental: only re-index changed sections (content hash based)
  - Non-destructive: never deletes user-created memories
  - Idempotent: running twice with same content = no-op
  - Lightweight: runs in <1s for typical workspace sizes
  - Source-tagged: all synced memories carry source="workspace-sync:{filename}"
    so they can be identified and cleaned up on re-sync
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

logger = logging.getLogger(__name__)

# Files that should be synced to Memory Store for vector search
_SYNCABLE_FILES = {"MEMORY.md"}

# Source tag prefix for workspace-synced memories
_SOURCE_PREFIX = "workspace-sync"


def _content_hash(text: str) -> str:
    """Deterministic hash for content dedup."""
    return hashlib.sha256(text.strip().encode()).hexdigest()[:16]


def _split_sections(content: str, max_chars: int = 4000) -> list[str]:
    """Split markdown by ## headers, further split oversized sections.

    Same logic as import_.base.split_markdown_sections but inlined
    to avoid circular imports and keep this module self-contained.
    """
    sections = re.split(r'^(## .+)$', content, flags=re.MULTILINE)
    result: list[str] = []
    current = ""

    for part in sections:
        if part.startswith("## "):
            if current.strip():
                result.append(current.strip())
            current = part + "\n"
        else:
            current += part

    if current.strip():
        result.append(current.strip())

    # Split oversized sections by paragraph
    final: list[str] = []
    for section in result:
        if len(section) <= max_chars:
            final.append(section)
        else:
            paragraphs = section.split("\n\n")
            chunk = ""
            for para in paragraphs:
                if len(chunk) + len(para) > max_chars and chunk:
                    final.append(chunk.strip())
                    chunk = ""
                chunk += para + "\n\n"
            if chunk.strip():
                final.append(chunk.strip())

    return [s for s in final if len(s.strip()) >= 20]


def _infer_type(text: str) -> MemoryType:
    """Lightweight type inference for workspace content."""
    text_lower = text[:500].lower()
    if any(w in text_lower for w in ["step", "how to", "install", "deploy", "run "]):
        return MemoryType.PROCEDURAL
    if any(w in text_lower for w in ["prefer", "like", "dislike", "style"]):
        return MemoryType.WORKING
    if re.search(r'20\d{2}-\d{2}-\d{2}', text_lower):
        return MemoryType.EPISODIC
    return MemoryType.SEMANTIC


class WorkspaceMemorySync:
    """Syncs workspace files (MEMORY.md etc.) to the Memory Store.

    Tracks what's been synced via a manifest file to enable incremental updates.
    On each sync:
      1. Read workspace files
      2. Split into sections
      3. Compare content hashes with last sync
      4. Delete stale sections (content changed or removed)
      5. Insert new/changed sections
    """

    def __init__(self, caveman_home: Path, memory_manager: Any):
        self.caveman_home = caveman_home
        self.memory_manager = memory_manager
        self._manifest_path = caveman_home / "workspace" / ".sync-manifest.json"
        self._workspace_paths = [
            caveman_home / "workspace",
            Path("~/.openclaw/workspace").expanduser(),
        ]

    def _load_manifest(self) -> dict:
        """Load the sync manifest (tracks what's been synced)."""
        if self._manifest_path.is_file():
            try:
                return json.loads(self._manifest_path.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError) as exc:
                logger.debug("Failed to load sync manifest: %s", exc)
        return {"version": 1, "files": {}}

    def _save_manifest(self, manifest: dict) -> None:
        """Persist the sync manifest."""
        self._manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest["last_sync"] = datetime.now().isoformat()
        self._manifest_path.write_text(
            json.dumps(manifest, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )

    def _find_workspace_file(self, filename: str) -> Path | None:
        """Find a workspace file across search paths (first wins)."""
        for ws_dir in self._workspace_paths:
            fp = ws_dir / filename
            if fp.is_file():
                return fp
        return None

    async def sync(self) -> dict:
        """Run incremental sync. Returns stats dict.

        Safe to call on every session start — fast no-op if nothing changed.
        """
        manifest = self._load_manifest()
        stats = {"checked": 0, "inserted": 0, "deleted": 0, "unchanged": 0}

        for filename in _SYNCABLE_FILES:
            fp = self._find_workspace_file(filename)
            if not fp:
                # File removed — clean up its memories
                if filename in manifest["files"]:
                    deleted = await self._delete_file_memories(filename, manifest)
                    stats["deleted"] += deleted
                    del manifest["files"][filename]
                continue

            stats["checked"] += 1
            content = fp.read_text(encoding="utf-8")
            file_hash = _content_hash(content)

            prev = manifest["files"].get(filename, {})
            if prev.get("file_hash") == file_hash:
                stats["unchanged"] += 1
                continue

            # File changed — incremental section-level sync
            sections = _split_sections(content)
            new_hashes: dict[str, str] = {}  # hash → section content
            for section in sections:
                h = _content_hash(section)
                new_hashes[h] = section

            old_hashes = set(prev.get("section_hashes", {}).keys())
            new_hash_set = set(new_hashes.keys())

            # Delete removed/changed sections
            to_delete = old_hashes - new_hash_set
            if to_delete:
                old_ids = prev.get("section_hashes", {})
                for h in to_delete:
                    mid = old_ids.get(h)
                    if mid:
                        await self._delete_memory(mid)
                        stats["deleted"] += 1

            # Insert new sections
            section_id_map: dict[str, str] = {}
            # Carry over unchanged section IDs
            for h in old_hashes & new_hash_set:
                old_id = prev.get("section_hashes", {}).get(h)
                if old_id:
                    section_id_map[h] = old_id

            to_insert = new_hash_set - old_hashes
            for h in to_insert:
                section = new_hashes[h]
                mem_type = _infer_type(section)
                source_tag = f"{_SOURCE_PREFIX}:{filename}"
                try:
                    mid = await self.memory_manager.store(
                        section, mem_type,
                        metadata={
                            "source": source_tag,
                            "source_file": str(fp),
                            "content_hash": h,
                            "synced_at": datetime.now().isoformat(),
                        },
                    )
                    if mid:
                        section_id_map[h] = mid
                        stats["inserted"] += 1
                except Exception as e:
                    logger.warning("Failed to sync section from %s: %s", filename, e)

            # Update manifest
            manifest["files"][filename] = {
                "file_hash": file_hash,
                "section_hashes": section_id_map,
                "path": str(fp),
                "section_count": len(sections),
            }

        self._save_manifest(manifest)

        if stats["inserted"] or stats["deleted"]:
            logger.info(
                "WorkspaceMemorySync: checked=%d inserted=%d deleted=%d unchanged=%d",
                stats["checked"], stats["inserted"], stats["deleted"], stats["unchanged"],
            )
        else:
            logger.debug("WorkspaceMemorySync: no changes detected")

        return stats

    async def _delete_file_memories(self, filename: str, manifest: dict) -> int:
        """Delete all memories synced from a specific file."""
        prev = manifest["files"].get(filename, {})
        section_hashes = prev.get("section_hashes", {})
        deleted = 0
        for h, mid in section_hashes.items():
            if mid:
                await self._delete_memory(mid)
                deleted += 1
        return deleted

    async def _delete_memory(self, memory_id: str) -> None:
        """Delete a single memory by ID."""
        try:
            if hasattr(self.memory_manager, 'delete'):
                await self.memory_manager.delete(memory_id)
            elif hasattr(self.memory_manager, '_backend') and self.memory_manager._backend:
                backend = self.memory_manager._backend
                if hasattr(backend, 'delete'):
                    await backend.delete(memory_id)
                else:
                    # Direct SQLite delete as last resort
                    conn = backend._get_conn()
                    conn.execute("DELETE FROM memories WHERE id = ?", (memory_id,))
                    conn.execute("DELETE FROM embeddings WHERE memory_id = ?", (memory_id,))
                    conn.commit()
        except Exception as e:
            logger.debug("Failed to delete memory %s: %s", memory_id, e)

    async def force_resync(self) -> dict:
        """Force full re-sync by clearing the manifest first."""
        # Delete all workspace-synced memories
        manifest = self._load_manifest()
        for filename in list(manifest.get("files", {}).keys()):
            await self._delete_file_memories(filename, manifest)
        # Clear manifest
        manifest = {"version": 1, "files": {}}
        self._save_manifest(manifest)
        # Run fresh sync
        return await self.sync()
