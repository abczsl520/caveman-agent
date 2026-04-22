"""Checkpoint Manager — save and restore agent state.

Provides checkpoint/restore for agent sessions, enabling
recovery from crashes and session migration. Extracted from
Hermes tools/checkpoint_manager.py (623 lines).
"""
from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = ["Checkpoint", "CheckpointManager"]


logger = logging.getLogger("caveman.tools.checkpoint")

_CHECKPOINT_DIR = Path.home() / ".caveman" / "checkpoints"


@dataclass
class Checkpoint:
    """A saved agent state checkpoint."""
    id: str
    session_key: str
    created_at: float = 0
    messages: List[Dict[str, Any]] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    model: str = ""
    token_count: int = 0
    description: str = ""

    @property
    def age_seconds(self) -> float:
        return time.time() - self.created_at if self.created_at else 0


class CheckpointManager:
    """Manages agent state checkpoints."""

    def __init__(self, base_dir: Optional[Path] = None, max_per_session: int = 5):
        self._base_dir = base_dir or _CHECKPOINT_DIR
        self._max_per_session = max_per_session

    def save(
        self,
        session_key: str,
        messages: List[Dict[str, Any]],
        metadata: Optional[Dict[str, Any]] = None,
        model: str = "",
        description: str = "",
    ) -> Checkpoint:
        """Save a checkpoint."""
        import uuid
        checkpoint_id = f"{int(time.time())}_{uuid.uuid4().hex[:12]}_{session_key.replace(':', '_')[:20]}"
        checkpoint = Checkpoint(
            id=checkpoint_id,
            session_key=session_key,
            created_at=time.time(),
            messages=messages,
            metadata=metadata or {},
            model=model,
            token_count=sum(len(str(m.get("content", ""))) // 4 for m in messages),
            description=description,
        )

        # Save to disk
        session_dir = self._session_dir(session_key)
        session_dir.mkdir(parents=True, exist_ok=True)
        path = session_dir / f"{checkpoint_id}.json"
        path.write_text(
            json.dumps({
                "id": checkpoint.id,
                "session_key": checkpoint.session_key,
                "created_at": checkpoint.created_at,
                "messages": checkpoint.messages,
                "metadata": checkpoint.metadata,
                "model": checkpoint.model,
                "token_count": checkpoint.token_count,
                "description": checkpoint.description,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        # Enforce max checkpoints per session
        self._enforce_limit(session_key)

        return checkpoint

    def restore(self, session_key: str, checkpoint_id: Optional[str] = None) -> Optional[Checkpoint]:
        """Restore a checkpoint. If no ID given, restore the latest."""
        session_dir = self._session_dir(session_key)
        if not session_dir.exists():
            return None

        if checkpoint_id:
            path = session_dir / f"{checkpoint_id}.json"
            if path.exists():
                return self._load_checkpoint(path)
            return None

        # Find latest
        checkpoints = sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)
        if checkpoints:
            return self._load_checkpoint(checkpoints[0])
        return None

    def list_checkpoints(self, session_key: str) -> List[Dict[str, Any]]:
        """List checkpoints for a session."""
        session_dir = self._session_dir(session_key)
        if not session_dir.exists():
            return []

        results = []
        for path in sorted(session_dir.glob("*.json"), key=lambda p: -p.stat().st_mtime):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
                results.append({
                    "id": data.get("id", path.stem),
                    "created_at": data.get("created_at", 0),
                    "model": data.get("model", ""),
                    "token_count": data.get("token_count", 0),
                    "message_count": len(data.get("messages", [])),
                    "description": data.get("description", ""),
                })
            except Exception as exc:
                logger.debug("list_checkpoints: suppressed %s", exc)
        return results

    def delete(self, session_key: str, checkpoint_id: str) -> bool:
        """Delete a specific checkpoint."""
        path = self._session_dir(session_key) / f"{checkpoint_id}.json"
        if path.exists():
            path.unlink(missing_ok=True)
            return True
        return False

    def cleanup(self, max_age_hours: int = 168) -> int:
        """Remove old checkpoints across all sessions."""
        if not self._base_dir.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        for session_dir in self._base_dir.iterdir():
            if not session_dir.is_dir():
                continue
            for path in session_dir.glob("*.json"):
                try:
                    if path.stat().st_mtime < cutoff:
                        path.unlink(missing_ok=True)
                        removed += 1
                except Exception:
                    pass  # intentional: Exception suppressed
            # Remove empty dirs
            if not any(session_dir.iterdir()):
                session_dir.rmdir()
        return removed

    def _session_dir(self, session_key: str) -> Path:
        safe_key = session_key.replace(":", "_").replace("/", "_")
        return self._base_dir / safe_key

    def _load_checkpoint(self, path: Path) -> Optional[Checkpoint]:
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            return Checkpoint(**data)
        except Exception as e:
            logger.debug("Failed to load checkpoint %s: %s", path, e)
            return None

    def _enforce_limit(self, session_key: str) -> None:
        session_dir = self._session_dir(session_key)
        checkpoints = sorted(session_dir.glob("*.json"), key=lambda p: p.stat().st_mtime)
        while len(checkpoints) > self._max_per_session:
            oldest = checkpoints.pop(0)
            oldest.unlink(missing_ok=True)