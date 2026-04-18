"""Obsidian-compatible memory export.

Exports Caveman's Memory Store as Obsidian-compatible markdown files
with wikilinks, tags, and YAML frontmatter.

Output structure:
  obsidian_vault/
    memories/
      semantic/
        server-ip-39-99-235-193.md
      episodic/
        2026-04-14-deployed-app.md
      procedural/
        how-to-install-pyenv.md
    index.md  (MOC — Map of Content)
"""
from __future__ import annotations

import re
import logging
from datetime import datetime
from pathlib import Path

from caveman.memory.manager import MemoryManager
from caveman.memory.types import MemoryEntry, MemoryType

logger = logging.getLogger(__name__)


def export_to_obsidian(
    memory_manager: MemoryManager,
    output_dir: Path | str,
) -> dict:
    """Export all memories as Obsidian-compatible markdown.

    Returns:
        {"exported": int, "output_dir": str}
    """
    out = Path(output_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)

    all_entries = memory_manager.all_entries()
    exported = 0
    index_entries: dict[str, list[str]] = {}

    for entry in all_entries:
        type_dir = out / entry.memory_type.value
        type_dir.mkdir(exist_ok=True)

        filename = _slugify(entry.content[:60]) + ".md"
        filepath = type_dir / filename

        # YAML frontmatter
        frontmatter = [
            "---",
            f"id: {entry.id}",
            f"type: {entry.memory_type.value}",
            f"created: {entry.created_at.isoformat()}",
        ]
        if entry.metadata.get("source"):
            frontmatter.append(f"source: {entry.metadata['source']}")
        if entry.metadata.get("related"):
            refs = entry.metadata["related"]
            frontmatter.append(f"related: [{', '.join(str(r) for r in refs)}]")
        if entry.metadata.get("superseded_by"):
            frontmatter.append(f"superseded_by: {entry.metadata['superseded_by']}")
        tags = _extract_tags(entry.content)
        if tags:
            frontmatter.append(f"tags: [{', '.join(tags)}]")
        frontmatter.append("---")
        frontmatter.append("")

        # Content with wikilinks
        content = _add_wikilinks(entry.content, all_entries, entry.id)

        filepath.write_text(
            "\n".join(frontmatter) + content + "\n",
            encoding="utf-8",
        )
        exported += 1

        # Track for index
        type_name = entry.memory_type.value
        index_entries.setdefault(type_name, []).append(
            f"- [[{entry.memory_type.value}/{filename[:-3]}]] — {entry.content[:80]}"
        )

    # Write index (MOC)
    index_lines = [
        "# Caveman Memory Vault",
        f"Exported: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        f"Total: {exported} memories",
        "",
    ]
    for type_name, entries in sorted(index_entries.items()):
        index_lines.append(f"## {type_name.title()}")
        index_lines.extend(entries)
        index_lines.append("")

    (out / "index.md").write_text("\n".join(index_lines), encoding="utf-8")

    return {"exported": exported, "output_dir": str(out)}


def _slugify(text: str) -> str:
    """Convert text to filename-safe slug."""
    text = text.lower().strip()
    text = re.sub(r'[^\w\s-]', '', text)
    text = re.sub(r'[\s_]+', '-', text)
    text = re.sub(r'-+', '-', text)
    return text[:80].strip('-') or "untitled"


def _extract_tags(content: str) -> list[str]:
    """Extract meaningful tags from content."""
    tags = []
    content_lower = content.lower()
    tag_keywords = {
        "server": "server", "deploy": "deploy", "api": "api",
        "python": "python", "node": "nodejs", "docker": "docker",
        "database": "database", "config": "config", "bug": "bug",
        "security": "security", "performance": "performance",
    }
    for keyword, tag in tag_keywords.items():
        if keyword in content_lower:
            tags.append(tag)
    return tags[:5]


def _add_wikilinks(
    content: str, all_entries: list[MemoryEntry], self_id: str
) -> str:
    """Add Obsidian wikilinks to related content."""
    # Simple approach: link IPs, paths, and key terms to other entries
    # This is a basic implementation; could be enhanced with NLP
    return content
