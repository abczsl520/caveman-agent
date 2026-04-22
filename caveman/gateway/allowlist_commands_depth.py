"""Allowlist Commands Depth — bulk ops, pattern matching, import/export.

Supplements allowlist_commands.py with bulk operations and
pattern-based matching. Extracted from OpenClaw commands-allowlist.ts.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

__all__ = [
    "AllowlistRule",
    "bulk_add",
    "bulk_remove",
    "export_allowlist",
    "import_allowlist",
    "cleanup_expired",
]


logger = logging.getLogger("caveman.gateway.allowlist_commands_depth")


@dataclass
class AllowlistRule:
    """An allowlist rule with pattern support."""
    pattern: str
    label: str = ""
    added_at: float = 0
    added_by: str = ""
    expires_at: float = 0  # 0 = never
    tags: List[str] = field(default_factory=list)

    @property
    def is_expired(self) -> bool:
        return self.expires_at > 0 and time.time() > self.expires_at

    def matches(self, identifier: str) -> bool:
        """Check if identifier matches this rule."""
        if self.is_expired:
            return False
        if "*" in self.pattern or "?" in self.pattern:
            return fnmatch.fnmatch(identifier, self.pattern)
        return self.pattern == identifier


def bulk_add(
    rules: List[AllowlistRule],
    existing: List[AllowlistRule],
) -> Dict[str, Any]:
    """Bulk add rules, skipping duplicates."""
    existing_patterns: Set[str] = {r.pattern for r in existing}
    added = []
    skipped = []
    for rule in rules:
        if rule.pattern in existing_patterns:
            skipped.append(rule.pattern)
        else:
            existing.append(rule)
            existing_patterns.add(rule.pattern)
            added.append(rule.pattern)
    return {"added": added, "skipped": skipped}


def bulk_remove(
    patterns: List[str],
    existing: List[AllowlistRule],
) -> Dict[str, Any]:
    """Bulk remove rules by pattern."""
    to_remove = set(patterns)
    removed = []
    remaining = []
    for rule in existing:
        if rule.pattern in to_remove:
            removed.append(rule.pattern)
        else:
            remaining.append(rule)
    existing.clear()
    existing.extend(remaining)
    return {"removed": removed, "not_found": list(to_remove - set(removed))}


def export_allowlist(rules: List[AllowlistRule], output_path: Optional[Path] = None) -> str:
    """Export allowlist to JSON."""
    data = [
        {
            "pattern": r.pattern,
            "label": r.label,
            "added_at": r.added_at,
            "added_by": r.added_by,
            "expires_at": r.expires_at,
            "tags": r.tags,
        }
        for r in rules
    ]
    text = json.dumps(data, ensure_ascii=False, indent=2)
    if output_path:
        output_path.write_text(text, encoding="utf-8")
    return text


def import_allowlist(data: str) -> List[AllowlistRule]:
    """Import allowlist from JSON."""
    items = json.loads(data)
    return [
        AllowlistRule(
            pattern=item["pattern"],
            label=item.get("label", ""),
            added_at=item.get("added_at", time.time()),
            added_by=item.get("added_by", ""),
            expires_at=item.get("expires_at", 0),
            tags=item.get("tags", []),
        )
        for item in items
    ]


def cleanup_expired(rules: List[AllowlistRule]) -> Dict[str, Any]:
    """Remove expired rules."""
    expired = [r for r in rules if r.is_expired]
    active = [r for r in rules if not r.is_expired]
    rules.clear()
    rules.extend(active)
    return {"removed": len(expired), "remaining": len(active)}
