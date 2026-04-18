"""Base classes for the import system — ABC, dataclasses, type inference."""
from __future__ import annotations

import hashlib
import json
import logging
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from caveman.memory.types import MemoryType

logger = logging.getLogger(__name__)


@dataclass
class ImportItem:
    """A single item to be imported."""

    source_path: Path
    target_type: str  # "memory" | "workspace" | "skill" | "config" | "cron"
    memory_type: MemoryType | None = None
    content: str = ""
    content_hash: str = ""
    size_bytes: int = 0
    preview: str = ""
    skip_reason: str | None = None

    def __post_init__(self) -> None:
        if self.content and not self.content_hash:
            self.content_hash = hashlib.sha256(self.content.encode()).hexdigest()[:16]
        if self.content and not self.size_bytes:
            self.size_bytes = len(self.content.encode())
        if self.content and not self.preview:
            self.preview = self.content[:100].replace("\n", " ")


@dataclass
class ImportManifest:
    """What will be imported from a source."""

    source: str
    items: list[ImportItem] = field(default_factory=list)

    @property
    def total_size(self) -> int:
        return sum(i.size_bytes for i in self.items)

    @property
    def actionable(self) -> list[ImportItem]:
        return [i for i in self.items if not i.skip_reason]

    @property
    def skipped(self) -> list[ImportItem]:
        return [i for i in self.items if i.skip_reason]

    @property
    def summary(self) -> str:
        act = len(self.actionable)
        skip = len(self.skipped)
        size_kb = self.total_size / 1024
        return f"{self.source}: {act} items ({size_kb:.1f} KB), {skip} skipped"


@dataclass
class ImportResult:
    """Result of an import execution."""

    imported: int = 0
    skipped: int = 0
    failed: int = 0
    duplicates: int = 0
    files_processed: int = 0
    details: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def summary(self) -> str:
        return (
            f"Import: {self.imported} imported, {self.duplicates} duplicates, "
            f"{self.skipped} skipped, {self.failed} failed "
            f"({self.files_processed} files)"
        )


class BaseImporter(ABC):
    """Abstract base for all source importers."""

    def __init__(self, caveman_home: Path, dry_run: bool = True, include_secrets: bool = False) -> None:
        self.caveman_home = caveman_home
        self.dry_run = dry_run
        self.include_secrets = include_secrets
        self.result = ImportResult()

    @property
    @abstractmethod
    def source_name(self) -> str:
        """Human-readable source name."""

    @abstractmethod
    def detect(self) -> bool:
        """Check if this source exists on the system."""

    @abstractmethod
    def scan(self) -> ImportManifest:
        """Scan source and return what would be imported (no writes)."""

    @abstractmethod
    async def execute(self, manifest: ImportManifest, memory_manager: Any) -> ImportResult:
        """Execute the import based on the manifest."""

    def _scan_secrets(self, content: str) -> str | None:
        """Scan content for secrets. Returns warning string or None.
        
        If include_secrets=True, returns None (allows all content through).
        """
        if self.include_secrets:
            return None
        try:
            from caveman.security.scanner import scan
            result = scan(content)
            if result.has_secrets:
                types = [name for name, _ in result.matches]
                return f"secrets detected: {types}"
        except Exception as e:
            logger.debug("Suppressed in base: %s", e)
        return None


def infer_type(text: str, source_path: Path) -> MemoryType:
    """Infer memory type from content + file path."""
    path_str = str(source_path).lower()

    # Path-based inference (highest priority)
    if "lessons" in path_str or "learnings" in path_str:
        return MemoryType.PROCEDURAL
    if "sop" in path_str:
        return MemoryType.PROCEDURAL
    if "projects" in path_str:
        return MemoryType.SEMANTIC
    if "archive" in path_str:
        return MemoryType.EPISODIC
    if "agent-profiles" in path_str:
        return MemoryType.SEMANTIC
    if "studies" in path_str:
        return MemoryType.SEMANTIC
    if "seo" in path_str:
        return MemoryType.SEMANTIC
    if re.search(r'\d{4}-\d{2}-\d{2}', path_str):
        return MemoryType.EPISODIC
    if "done/" in path_str or "done\\" in path_str:
        return MemoryType.EPISODIC

    # Content-based inference (fallback)
    text_lower = text[:500].lower()
    if any(w in text_lower for w in ["step", "how to", "install", "deploy", "run "]):
        return MemoryType.PROCEDURAL
    if any(w in text_lower for w in ["prefer", "like", "dislike", "style"]):
        return MemoryType.WORKING
    if re.search(r'20\d{2}-\d{2}-\d{2}', text_lower):
        return MemoryType.EPISODIC

    return MemoryType.SEMANTIC


def split_markdown_sections(content: str, max_chars: int = 4000) -> list[str]:
    """Split markdown by ## headers, further split oversized sections."""
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


def write_import_log(caveman_home: Path, entry: dict) -> None:
    """Append an entry to ~/.caveman/import-log.jsonl."""
    log_path = caveman_home / "import-log.jsonl"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    entry["timestamp"] = datetime.now().isoformat()
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")
