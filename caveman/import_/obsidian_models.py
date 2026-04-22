"""Obsidian Vault Bridge — live sync between Obsidian and Caveman's memory.

Architecture decision (长期主义 + 最高复利):

  NOT a one-time import. That's "够用就行" thinking.

  Instead: Obsidian vault becomes a **live knowledge source** for Caveman.
  - User edits in Obsidian → Caveman sees it next recall
  - Caveman learns something → can write back to Obsidian
  - File changes tracked incrementally (hash + mtime)
  - Obsidian front matter (tags, date, source) preserved and used for classification
  - Wikilinks [[note]] resolved to enrich context

  This means:
  1. User's existing Obsidian workflow is untouched
  2. Knowledge flows both ways
  3. No duplicate data — vault IS the source of truth for human-curated knowledge
  4. Caveman's vector index is a derived view, rebuilt from vault on demand

Design:
  - `ObsidianVault`: reads vault structure, parses front matter, resolves wikilinks
  - `VaultSyncState`: tracks file hashes for incremental sync (what changed since last sync)
  - `ObsidianBridge`: orchestrates sync — vault → memory store (index) and memory store → vault (write-back)
  - CLI: `caveman vault link <path>` / `caveman vault sync` / `caveman vault status`
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import yaml

from caveman.memory.types import MemoryType

__all__ = ["ObsidianNote", "parse_note", "ObsidianVault", "FileSyncRecord", "VaultSyncState", "SyncResult"]


logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Obsidian note parsing
# ---------------------------------------------------------------------------

@dataclass
class ObsidianNote:
    """A parsed Obsidian markdown note."""
    path: Path
    relative_path: str  # relative to vault root
    content: str
    front_matter: dict[str, Any] = field(default_factory=dict)
    body: str = ""  # content without front matter
    tags: list[str] = field(default_factory=list)
    wikilinks: list[str] = field(default_factory=list)
    file_hash: str = ""
    mtime: float = 0.0
    size: int = 0

    @property
    def title(self) -> str:
        """Note title from first H1 or filename."""
        for line in self.body.split("\n"):
            line = line.strip()
            if line.startswith("# ") and not line.startswith("## "):
                return line[2:].strip()
        return self.path.stem

    @property
    def date(self) -> str:
        """Date from front matter or filename."""
        if "date" in self.front_matter:
            return str(self.front_matter["date"])[:10]
        # Try YYYY-MM-DD in filename
        m = re.search(r"(\d{4}-\d{2}-\d{2})", self.path.stem)
        return m.group(1) if m else ""

    @property
    def folder(self) -> str:
        """Parent folder name (used for category inference)."""
        parts = Path(self.relative_path).parts
        return parts[0] if len(parts) > 1 else ""

    @property
    def inferred_memory_type(self) -> MemoryType:
        """Infer memory type from folder, tags, and content."""
        folder_lower = self.folder.lower()
        tags_lower = [t.lower() for t in self.tags]

        # Folder-based
        folder_map = {
            "经验教训": MemoryType.PROCEDURAL,
            "教训": MemoryType.PROCEDURAL,
            "lessons": MemoryType.PROCEDURAL,
            "运维": MemoryType.PROCEDURAL,
            "日常": MemoryType.EPISODIC,
            "daily": MemoryType.EPISODIC,
            "日记": MemoryType.EPISODIC,
            "journal": MemoryType.EPISODIC,
            "项目": MemoryType.SEMANTIC,
            "projects": MemoryType.SEMANTIC,
            "架构": MemoryType.SEMANTIC,
            "architecture": MemoryType.SEMANTIC,
            "工具": MemoryType.SEMANTIC,
            "tools": MemoryType.SEMANTIC,
            "服务器": MemoryType.SEMANTIC,
            "servers": MemoryType.SEMANTIC,
        }
        for key, mtype in folder_map.items():
            if key in folder_lower:
                return mtype

        # Tag-based
        if any(t in tags_lower for t in ["教训", "踩坑", "gotcha", "lesson", "sop"]):
            return MemoryType.PROCEDURAL
        if any(t in tags_lower for t in ["架构", "architecture", "设计", "竞品"]):
            return MemoryType.SEMANTIC
        if any(t in tags_lower for t in ["日记", "daily", "journal"]):
            return MemoryType.EPISODIC

        return MemoryType.SEMANTIC


def parse_note(path: Path, vault_root: Path) -> ObsidianNote | None:
    """Parse an Obsidian markdown file."""
    try:
        raw = path.read_text(encoding="utf-8")
    except Exception as e:
        logger.debug("Failed to read %s: %s", path, e)
        return None

    if not raw.strip():
        return None

    stat = path.stat()
    file_hash = hashlib.sha256(raw.encode()).hexdigest()[:16]
    relative = str(path.relative_to(vault_root))

    # Parse front matter
    front_matter: dict[str, Any] = {}
    body = raw
    if raw.startswith("---"):
        end = raw.find("---", 3)
        if end > 0:
            fm_text = raw[3:end].strip()
            try:
                front_matter = yaml.safe_load(fm_text) or {}
            except yaml.YAMLError as exc:
                logger.debug("parse_note: suppressed %s", exc)
            body = raw[end + 3:].strip()

    # Also check HTML comment tags: <!-- tags: a, b, c -->
    comment_tags = re.findall(r"<!--\s*tags?:\s*(.+?)\s*-->", raw)

    # Extract tags
    tags: list[str] = []
    if "tags" in front_matter:
        fm_tags = front_matter["tags"]
        if isinstance(fm_tags, list):
            tags.extend(str(t) for t in fm_tags)
        elif isinstance(fm_tags, str):
            tags.extend(t.strip() for t in fm_tags.split(","))
    for ct in comment_tags:
        tags.extend(t.strip() for t in ct.split(","))

    # Extract wikilinks
    wikilinks = re.findall(r"\[\[([^\]|]+?)(?:\|[^\]]+?)?\]\]", raw)

    return ObsidianNote(
        path=path,
        relative_path=relative,
        content=raw,
        front_matter=front_matter,
        body=body,
        tags=tags,
        wikilinks=wikilinks,
        file_hash=file_hash,
        mtime=stat.st_mtime,
        size=stat.st_size,
    )


# ---------------------------------------------------------------------------
# Vault scanner
# ---------------------------------------------------------------------------

@dataclass
class ObsidianVault:
    """Represents an Obsidian vault on disk."""
    root: Path
    notes: dict[str, ObsidianNote] = field(default_factory=dict)  # relative_path → note

    @property
    def note_count(self) -> int:
        return len(self.notes)

    @property
    def total_size(self) -> int:
        return sum(n.size for n in self.notes.values())

    def scan(self) -> None:
        """Scan vault for all markdown files."""
        self.notes.clear()
        for md in sorted(self.root.rglob("*.md")):
            # Skip .obsidian config dir and hidden files
            rel = md.relative_to(self.root)
            if any(p.startswith(".") for p in rel.parts):
                continue
            note = parse_note(md, self.root)
            if note:
                self.notes[note.relative_path] = note

    def resolve_wikilink(self, link_text: str) -> ObsidianNote | None:
        """Resolve a [[wikilink]] to a note."""
        # Exact match by stem
        for note in self.notes.values():
            if note.path.stem == link_text:
                return note
        # Fuzzy match
        link_lower = link_text.lower()
        for note in self.notes.values():
            if note.path.stem.lower() == link_lower:
                return note
        return None


# ---------------------------------------------------------------------------
# Sync state tracking
# ---------------------------------------------------------------------------

_SYNC_STATE_FILE = "obsidian-sync-state.json"


@dataclass
class FileSyncRecord:
    """Tracking record for one synced file."""
    relative_path: str
    file_hash: str
    mtime: float
    memory_ids: list[str] = field(default_factory=list)  # IDs in memory store
    synced_at: str = ""


@dataclass
class VaultSyncState:
    """Persistent state for incremental vault sync."""
    vault_path: str = ""
    last_sync: str = ""
    files: dict[str, FileSyncRecord] = field(default_factory=dict)

    def needs_sync(self, note: ObsidianNote) -> bool:
        """Check if a note has changed since last sync."""
        rec = self.files.get(note.relative_path)
        if rec is None:
            return True  # New file
        return rec.file_hash != note.file_hash

    def mark_synced(self, note: ObsidianNote, memory_ids: list[str]) -> None:
        """Record that a note has been synced."""
        self.files[note.relative_path] = FileSyncRecord(
            relative_path=note.relative_path,
            file_hash=note.file_hash,
            mtime=note.mtime,
            memory_ids=memory_ids,
            synced_at=datetime.now().isoformat(),
        )
        self.last_sync = datetime.now().isoformat()

    def get_deleted(self, current_notes: set[str]) -> list[FileSyncRecord]:
        """Find files that were synced before but no longer exist."""
        return [
            rec for path, rec in self.files.items()
            if path not in current_notes
        ]

    def save(self, caveman_home: Path) -> None:
        """Persist sync state to disk."""
        state_path = caveman_home / _SYNC_STATE_FILE
        data = {
            "vault_path": self.vault_path,
            "last_sync": self.last_sync,
            "files": {
                path: {
                    "relative_path": rec.relative_path,
                    "file_hash": rec.file_hash,
                    "mtime": rec.mtime,
                    "memory_ids": rec.memory_ids,
                    "synced_at": rec.synced_at,
                }
                for path, rec in self.files.items()
            },
        }
        state_path.write_text(
            json.dumps(data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, caveman_home: Path) -> "VaultSyncState":
        """Load sync state from disk."""
        state_path = caveman_home / _SYNC_STATE_FILE
        if not state_path.exists():
            return cls()
        try:
            data = json.loads(state_path.read_text(encoding="utf-8"))
            state = cls(
                vault_path=data.get("vault_path", ""),
                last_sync=data.get("last_sync", ""),
            )
            for path, rec_data in data.get("files", {}).items():
                state.files[path] = FileSyncRecord(**rec_data)
            return state
        except Exception as e:
            logger.warning("Failed to load sync state: %s", e)
            return cls()


# ---------------------------------------------------------------------------
# Obsidian Bridge — the main orchestrator
# ---------------------------------------------------------------------------

@dataclass
class SyncResult:
    """Result of a vault sync operation."""
    added: int = 0
    updated: int = 0
    deleted: int = 0
    unchanged: int = 0
    failed: int = 0
    errors: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Sync: +{self.added} added, ~{self.updated} updated, "
            f"-{self.deleted} deleted, ={self.unchanged} unchanged, "
            f"✗{self.failed} failed"
        )


def _split_note_for_memory(note: ObsidianNote, max_chars: int = 3000) -> list[str]:
    """Split a note into memory-sized chunks.

    Strategy:
    - Split by ## headers first
    - Each chunk gets a context header with note title + tags
    - Oversized sections split by paragraph
    - Short notes kept as single entry
    """
    body = note.body
    if not body.strip():
        return []

    # Context header for each chunk
    header_parts = [f"[Obsidian] {note.title}"]
    if note.tags:
        header_parts.append(f"tags: {', '.join(note.tags[:8])}")
    if note.date:
        header_parts.append(f"date: {note.date}")
    if note.folder:
        header_parts.append(f"folder: {note.folder}")
    context_header = " | ".join(header_parts)

    # If note is small enough, keep as one entry
    if len(body) <= max_chars:
        return [f"{context_header}\n\n{body}"]

    # Split by ## headers
    sections = re.split(r"^(## .+)$", body, flags=re.MULTILINE)
    chunks: list[str] = []
    current = ""

    for part in sections:
        if part.startswith("## "):
            if current.strip() and len(current.strip()) >= 30:
                chunks.append(current.strip())
            current = part + "\n"
        else:
            current += part

    if current.strip() and len(current.strip()) >= 30:
        chunks.append(current.strip())

    # Further split oversized chunks
    final: list[str] = []
    for chunk in chunks:
        if len(chunk) <= max_chars:
            final.append(f"{context_header}\n\n{chunk}")
        else:
            # Split by paragraph
            paragraphs = chunk.split("\n\n")
            buf = ""
            for para in paragraphs:
                if len(buf) + len(para) > max_chars and buf:
                    final.append(f"{context_header}\n\n{buf.strip()}")
                    buf = ""
                buf += para + "\n\n"
            if buf.strip() and len(buf.strip()) >= 30:
                final.append(f"{context_header}\n\n{buf.strip()}")

    return final if final else [f"{context_header}\n\n{body[:max_chars]}"]

