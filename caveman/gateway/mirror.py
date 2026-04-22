"""Session mirroring for cross-platform message delivery.

When a message is sent via CLI, cron, or a different gateway, this module
appends a "delivery-mirror" record to the target session's transcript so
the receiving-side agent has context about what was sent.

Works standalone — no full SessionStore needed.
"""
from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

_SESSIONS_DIR = CAVEMAN_HOME / "sessions"


def mirror_to_session(
    platform: str,
    chat_id: str,
    message_text: str,
    source_label: str = "cli",
    thread_id: str | None = None,
    sessions_dir: Path | None = None,
) -> bool:
    """Append a delivery-mirror message to the target session's transcript.

    Finds the gateway session matching platform + chat_id, then writes
    a mirror entry to the JSONL transcript and SQLite DB.

    Returns True if mirrored, False if no session found or error.
    Never raises — all errors are caught.
    """
    sdir = sessions_dir or _SESSIONS_DIR
    try:
        session_id = _find_session_id(platform, str(chat_id), thread_id=thread_id, sessions_dir=sdir)
        if not session_id:
            logger.debug("Mirror: no session for %s:%s:%s", platform, chat_id, thread_id)
            return False

        mirror_msg = {
            "role": "assistant",
            "content": message_text,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "mirror": True,
            "mirror_source": source_label,
        }

        _append_to_jsonl(session_id, mirror_msg, sessions_dir=sdir)
        _append_to_sqlite(session_id, mirror_msg)

        logger.debug("Mirror: wrote to session %s (from %s)", session_id, source_label)
        return True

    except Exception as e:
        logger.debug("Mirror failed for %s:%s:%s: %s", platform, chat_id, thread_id, e)
        return False


def _find_session_id(
    platform: str,
    chat_id: str,
    thread_id: str | None = None,
    sessions_dir: Path | None = None,
) -> str | None:
    """Find the active session_id for a platform + chat_id pair."""
    sdir = sessions_dir or _SESSIONS_DIR
    index_path = sdir / "sessions.json"
    if not index_path.exists():
        return None

    try:
        data = json.loads(index_path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.debug("suppressed: %s", e)
        return None

    platform_lower = platform.lower()
    best_match = None
    best_updated = ""

    for _key, entry in data.items():
        origin = entry.get("origin") or {}
        entry_platform = (origin.get("platform") or entry.get("platform", "")).lower()

        if entry_platform != platform_lower:
            continue

        origin_chat_id = str(origin.get("chat_id", ""))
        if origin_chat_id != str(chat_id):
            continue

        if thread_id is not None:
            origin_thread = origin.get("thread_id")
            if str(origin_thread or "") != str(thread_id):
                continue

        updated = entry.get("updated_at", "")
        if updated > best_updated:
            best_updated = updated
            best_match = entry.get("session_id")

    return best_match


def _append_to_jsonl(session_id: str, message: dict[str, Any], sessions_dir: Path | None = None) -> None:
    """Append a message to the JSONL transcript file."""
    sdir = sessions_dir or _SESSIONS_DIR
    transcript_path = sdir / f"{session_id}.jsonl"
    try:
        with open(transcript_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(message, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Mirror JSONL write failed: %s", e)


def _append_to_sqlite(session_id: str, message: dict[str, Any]) -> None:
    """Append a message to the SQLite session database (if available)."""
    try:
        from caveman.agent.session_db import SessionDB
        db = SessionDB()
        db.append_transcript(
            session_id=session_id,
            role=message.get("role", "assistant"),
            content=message.get("content", ""),
        )
    except Exception as e:
        logger.debug("Mirror SQLite write failed: %s", e)
