"""Gateway channel directory — maps platform+chat_id to session metadata.

Provides a unified lookup for which sessions are active on which platforms,
enabling cross-platform features like session mirroring and message routing.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)


@dataclass
class ChannelEntry:
    """Registry entry mapping a platform channel to its metadata and routing config."""
    platform: str
    chat_id: str
    session_id: str
    display_name: str = ""
    chat_type: str = "dm"  # dm, group, channel
    thread_id: str = ""
    last_active: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ChannelDirectory:
    """Maps platform channels to sessions."""

    def __init__(self, data_dir: Path | None = None) -> None:
        self._dir = data_dir or (CAVEMAN_HOME / "gateway")
        self._index_file = self._dir / "channels.json"
        self._entries: dict[str, ChannelEntry] = {}
        self._load()

    def register(self, entry: ChannelEntry) -> None:
        """Register or update a channel entry."""
        key = self._key(entry.platform, entry.chat_id, entry.thread_id)
        entry.last_active = time.time()
        self._entries[key] = entry
        self._save()

    def lookup(self, platform: str, chat_id: str, thread_id: str = "") -> ChannelEntry | None:
        """Find a channel entry."""
        key = self._key(platform, chat_id, thread_id)
        return self._entries.get(key)

    def find_by_session(self, session_id: str) -> list[ChannelEntry]:
        """Find all channels for a session."""
        return [e for e in self._entries.values() if e.session_id == session_id]

    def find_by_platform(self, platform: str) -> list[ChannelEntry]:
        """Find all channels on a platform."""
        return [e for e in self._entries.values() if e.platform == platform]

    def remove(self, platform: str, chat_id: str, thread_id: str = "") -> bool:
        key = self._key(platform, chat_id, thread_id)
        if key in self._entries:
            del self._entries[key]
            self._save()
            return True
        return False

    def all_entries(self) -> list[ChannelEntry]:
        return list(self._entries.values())

    @staticmethod
    def _key(platform: str, chat_id: str, thread_id: str = "") -> str:
        parts = [platform.lower(), str(chat_id)]
        if thread_id:
            parts.append(str(thread_id))
        return ":".join(parts)

    def _load(self) -> None:
        if not self._index_file.exists():
            return
        try:
            data = json.loads(self._index_file.read_text(encoding="utf-8"))
            for key, entry_data in data.items():
                self._entries[key] = ChannelEntry(**entry_data)
        except Exception as e:
            logger.warning("Failed to load channel directory: %s", e)

    def _save(self) -> None:
        self._dir.mkdir(parents=True, exist_ok=True)
        data = {}
        for key, entry in self._entries.items():
            data[key] = {
                "platform": entry.platform,
                "chat_id": entry.chat_id,
                "session_id": entry.session_id,
                "display_name": entry.display_name,
                "chat_type": entry.chat_type,
                "thread_id": entry.thread_id,
                "last_active": entry.last_active,
                "metadata": entry.metadata,
            }
        self._index_file.write_text(json.dumps(data, indent=2), encoding="utf-8")
