"""Checkpoint system — save and restore agent state."""
from __future__ import annotations
import json
import logging
import re
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Only allow safe characters in session IDs and checkpoint IDs
_SAFE_ID_RE = re.compile(r"^[a-zA-Z0-9_\-]+$")


def _validate_id(value: str, label: str) -> None:
    """Reject IDs that could cause path traversal."""
    if not value or not _SAFE_ID_RE.match(value):
        raise ValueError(f"Invalid {label}: must be alphanumeric/underscore/hyphen, got {value!r}")


class Checkpoint:
    """Serializable snapshot of agent state."""

    def __init__(self, session_id: str, messages: list[dict], metadata: dict | None = None):
        self.session_id = session_id
        self.messages = messages
        self.metadata = metadata or {}
        self.created_at = datetime.now().isoformat()


class CheckpointManager:
    """Save/restore agent state to disk."""

    def __init__(self, base_dir: Path | None = None):
        self.base_dir = base_dir or Path.home() / ".caveman" / "checkpoints"
        self.base_dir.mkdir(parents=True, exist_ok=True)

    async def save(self, checkpoint: Checkpoint) -> str:
        """Save checkpoint, return checkpoint ID."""
        _validate_id(checkpoint.session_id, "session_id")
        cp_id = f"{checkpoint.session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        path = self.base_dir / f"{cp_id}.json"
        # Verify resolved path stays inside base_dir
        if not path.resolve().parent == self.base_dir.resolve():
            raise ValueError(f"Path traversal detected: {cp_id}")
        data = {
            "id": cp_id,
            "session_id": checkpoint.session_id,
            "messages": checkpoint.messages,
            "metadata": checkpoint.metadata,
            "created_at": checkpoint.created_at,
        }
        path.write_text(json.dumps(data, ensure_ascii=False, indent=2))
        await self._cleanup(checkpoint.session_id)
        return cp_id

    async def restore(self, session_id: str, cp_id: str | None = None) -> Checkpoint | None:
        """Restore latest or specific checkpoint."""
        _validate_id(session_id, "session_id")
        if cp_id:
            _validate_id(cp_id, "checkpoint_id")
            path = self.base_dir / f"{cp_id}.json"
        else:
            candidates = sorted(self.base_dir.glob(f"{session_id}_*.json"), reverse=True)
            if not candidates:
                return None
            path = candidates[0]
        # Verify resolved path stays inside base_dir
        if not path.resolve().parent == self.base_dir.resolve():
            raise ValueError(f"Path traversal detected")
        if not path.exists():
            return None
        try:
            data = json.loads(path.read_text())
        except (json.JSONDecodeError, OSError) as e:
            logger.warning("Corrupt checkpoint file %s: %s", path, e)
            return None
        cp = Checkpoint(data["session_id"], data["messages"], data.get("metadata"))
        cp.created_at = data.get("created_at", cp.created_at)
        return cp

    async def list_checkpoints(self, session_id: str | None = None) -> list[dict]:
        """List available checkpoints."""
        if session_id:
            _validate_id(session_id, "session_id")
        pattern = f"{session_id}_*.json" if session_id else "*.json"
        results = []
        for path in sorted(self.base_dir.glob(pattern), reverse=True):
            try:
                data = json.loads(path.read_text())
                results.append({
                    "id": data["id"],
                    "session_id": data["session_id"],
                    "created_at": data["created_at"],
                })
            except (json.JSONDecodeError, OSError, KeyError) as e:
                logger.warning("Skipping corrupt checkpoint %s: %s", path, e)
        return results

    async def _cleanup(self, session_id: str, keep: int = 10):
        candidates = sorted(self.base_dir.glob(f"{session_id}_*.json"))
        for old in candidates[:-keep]:
            try:
                old.unlink()
            except OSError as e:
                logger.warning("Failed to remove old checkpoint %s: %s", old, e)
