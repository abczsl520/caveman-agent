"""Allowlist Commands — user/channel/role access management.

Extracted from OpenClaw commands-allowlist.ts (573 lines).
Handles: /allowlist add, /allowlist remove, /allowlist list, pattern matching.
"""
from __future__ import annotations

import fnmatch
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

logger = logging.getLogger("caveman.gateway.allowlist_commands")


@dataclass
class AllowlistEntry:
    """An allowlist entry."""
    pattern: str  # user:123, channel:456, role:789, *
    added_by: str = ""
    added_at: float = 0
    reason: str = ""
    entry_type: str = ""  # user | channel | role | glob

    def matches(self, target: str) -> bool:
        """Check if this entry matches a target."""
        if self.pattern == "*":
            return True
        # Glob patterns (contain * or ?)
        if "*" in self.pattern or "?" in self.pattern:
            return fnmatch.fnmatch(target.lower(), self.pattern.lower())
        # Exact match
        return self.pattern == target


class AllowlistManager:
    """Manages allowlist for access control."""

    def __init__(self, persist_path: Optional[Path] = None):
        self._entries: Dict[str, AllowlistEntry] = {}
        self._persist_path = persist_path
        if persist_path and persist_path.exists():
            self._load()

    # ── Commands ──

    def add(self, pattern: str, added_by: str = "", reason: str = "") -> str:
        """Add an allowlist entry."""
        if pattern in self._entries:
            return f"Already in allowlist: {pattern}"

        entry_type = "glob"
        if pattern.startswith("user:"):
            entry_type = "user"
        elif pattern.startswith("channel:"):
            entry_type = "channel"
        elif pattern.startswith("role:"):
            entry_type = "role"

        self._entries[pattern] = AllowlistEntry(
            pattern=pattern,
            added_by=added_by,
            added_at=time.time(),
            reason=reason,
            entry_type=entry_type,
        )
        self._save()
        return f"Added to allowlist: {pattern}"

    def remove(self, pattern: str) -> str:
        """Remove an allowlist entry."""
        if pattern not in self._entries:
            return f"Not in allowlist: {pattern}"
        self._entries.pop(pattern)
        self._save()
        return f"Removed from allowlist: {pattern}"

    def list_entries(self, entry_type: str = "") -> List[AllowlistEntry]:
        """List allowlist entries."""
        entries = list(self._entries.values())
        if entry_type:
            entries = [e for e in entries if e.entry_type == entry_type]
        return sorted(entries, key=lambda e: e.pattern)

    def is_allowed(self, *targets: str) -> bool:
        """Check if any target matches the allowlist."""
        if not self._entries:
            return True  # Empty allowlist = allow all
        for target in targets:
            for entry in self._entries.values():
                if entry.matches(target):
                    return True
        return False

    def format_list(self) -> str:
        """Format allowlist for display."""
        entries = self.list_entries()
        if not entries:
            return "Allowlist is empty (all access allowed)."
        lines = ["Allowlist entries:"]
        for e in entries:
            added = time.strftime("%Y-%m-%d", time.localtime(e.added_at)) if e.added_at else "?"
            reason = f" — {e.reason}" if e.reason else ""
            lines.append(f"  {e.pattern} (by {e.added_by or '?'}, {added}){reason}")
        return "\n".join(lines)

    # ── Persistence ──

    def _save(self) -> None:
        if not self._persist_path:
            return
        self._persist_path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            k: {
                "pattern": v.pattern,
                "added_by": v.added_by,
                "added_at": v.added_at,
                "reason": v.reason,
                "entry_type": v.entry_type,
            }
            for k, v in self._entries.items()
        }
        self._persist_path.write_text(json.dumps(data, indent=2), encoding="utf-8")

    def _load(self) -> None:
        try:
            data = json.loads(self._persist_path.read_text(encoding="utf-8"))
            for k, v in data.items():
                self._entries[k] = AllowlistEntry(**v)
        except Exception as e:
            logger.warning("Failed to load allowlist: %s", e)

from caveman.gateway.allowlist_commands_depth import (  # noqa: F401,E402  # depth wiring
    AllowlistRule,
    bulk_add,
    bulk_remove,
    export_allowlist,
    import_allowlist,
    cleanup_expired,
)

__all__ = [
    "AllowlistEntry",
    "AllowlistManager",
    "AllowlistRule",
    "bulk_add",
    "bulk_remove",
    "export_allowlist",
    "import_allowlist",
    "cleanup_expired",
]
