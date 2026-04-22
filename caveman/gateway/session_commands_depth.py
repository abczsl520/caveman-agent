"""Session Commands Depth — export, import, search, bulk ops.

Supplements session_commands.py with session export/import,
search, and bulk operations. Extracted from OpenClaw commands-session.ts.
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "SessionExport",
    "export_session",
    "import_session",
    "search_sessions",
    "bulk_delete_sessions",
]


logger = logging.getLogger("caveman.gateway.session_commands_depth")


@dataclass
class SessionExport:
    """Exported session data."""
    session_key: str
    messages: List[Dict[str, Any]]
    metadata: Dict[str, Any]
    exported_at: float = 0

    def to_json(self) -> str:
        return json.dumps({
            "session_key": self.session_key,
            "messages": self.messages,
            "metadata": self.metadata,
            "exported_at": self.exported_at or time.time(),
        }, ensure_ascii=False, indent=2)

    @classmethod
    def from_json(cls, data: str) -> "SessionExport":
        d = json.loads(data)
        return cls(
            session_key=d["session_key"],
            messages=d["messages"],
            metadata=d.get("metadata", {}),
            exported_at=d.get("exported_at", 0),
        )


def export_session(
    session_key: str,
    messages: List[Dict[str, Any]],
    metadata: Optional[Dict[str, Any]] = None,
    output_dir: Optional[Path] = None,
) -> Path:
    """Export a session to a JSON file."""
    export = SessionExport(
        session_key=session_key,
        messages=messages,
        metadata=metadata or {},
        exported_at=time.time(),
    )
    out_dir = output_dir or Path.home() / ".caveman" / "exports"
    out_dir.mkdir(parents=True, exist_ok=True)
    safe_key = session_key.replace(":", "_").replace("/", "_")
    path = out_dir / f"session_{safe_key}_{int(time.time())}.json"
    path.write_text(export.to_json(), encoding="utf-8")
    return path


def import_session(path: Path) -> SessionExport:
    """Import a session from a JSON file."""
    data = path.read_text(encoding="utf-8")
    return SessionExport.from_json(data)


def search_sessions(
    sessions: List[Dict[str, Any]],
    query: str,
    limit: int = 20,
) -> List[Dict[str, Any]]:
    """Search sessions by key, model, or message content."""
    query_lower = query.lower()
    results = []
    for session in sessions:
        score = 0
        key = session.get("session_key", "").lower()
        model = session.get("model", "").lower()
        if query_lower in key:
            score += 10
        if query_lower in model:
            score += 5
        # Search in recent messages
        for msg in session.get("messages", [])[-10:]:
            if query_lower in str(msg.get("content", "")).lower():
                score += 1
        if score > 0:
            results.append({**session, "_score": score})
    results.sort(key=lambda x: -x["_score"])
    return results[:limit]


def bulk_delete_sessions(
    sessions: List[str],
    delete_fn: Any,
    dry_run: bool = False,
) -> Dict[str, Any]:
    """Bulk delete sessions."""
    deleted = []
    failed = []
    for key in sessions:
        if dry_run:
            deleted.append(key)
            continue
        try:
            result = delete_fn(key)
            if hasattr(result, "__await__"):
                import asyncio
                try:
                    loop = asyncio.get_running_loop()
                    loop.create_task(result)
                except RuntimeError:
                    asyncio.run(result)
            deleted.append(key)
        except Exception as e:
            failed.append({"key": key, "error": str(e)})
    return {"deleted": deleted, "failed": failed, "dry_run": dry_run}
