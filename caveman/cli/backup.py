"""Backup Manager — configuration and data backup.

Provides backup/restore for caveman configuration, sessions,
and memory. Extracted from Hermes hermes_cli/backup.py.
"""
from __future__ import annotations

import json
import logging
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger("caveman.cli.backup")

_BACKUP_DIR = Path.home() / ".caveman" / "backups"


@dataclass
class BackupManifest:
    """Manifest for a backup."""
    id: str
    created_at: float
    description: str = ""
    includes: List[str] = field(default_factory=list)
    size_bytes: int = 0
    path: str = ""


class BackupManager:
    """Manages backups of caveman data."""

    def __init__(self, home_dir: Optional[Path] = None, backup_dir: Optional[Path] = None):
        self._home = home_dir or Path.home() / ".caveman"
        self._backup_dir = backup_dir or _BACKUP_DIR

    def create(
        self,
        description: str = "",
        include: Optional[List[str]] = None,
    ) -> BackupManifest:
        """Create a backup."""
        backup_id = f"backup_{int(time.time())}"
        backup_path = self._backup_dir / backup_id
        backup_path.mkdir(parents=True, exist_ok=True)

        # Default includes
        targets = include or ["config", "memory", "skills", "sessions"]
        included = []
        total_size = 0

        for target in targets:
            source = self._resolve_source(target)
            if source and source.exists():
                dest = backup_path / target
                try:
                    if source.is_dir():
                        shutil.copytree(source, dest, dirs_exist_ok=True)
                    else:
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(source, dest)
                    included.append(target)
                    total_size += self._dir_size(dest) if dest.is_dir() else dest.stat().st_size
                except Exception as e:
                    logger.warning("Failed to backup %s: %s", target, e)

        manifest = BackupManifest(
            id=backup_id,
            created_at=time.time(),
            description=description,
            includes=included,
            size_bytes=total_size,
            path=str(backup_path),
        )

        # Save manifest
        (backup_path / "manifest.json").write_text(
            json.dumps({
                "id": manifest.id,
                "created_at": manifest.created_at,
                "description": manifest.description,
                "includes": manifest.includes,
                "size_bytes": manifest.size_bytes,
            }, ensure_ascii=False),
            encoding="utf-8",
        )

        return manifest

    def restore(self, backup_id: str, targets: Optional[List[str]] = None) -> Dict[str, Any]:
        """Restore from a backup."""
        backup_path = self._backup_dir / backup_id
        if not backup_path.exists():
            return {"success": False, "error": f"Backup not found: {backup_id}"}

        manifest_path = backup_path / "manifest.json"
        if not manifest_path.exists():
            return {"success": False, "error": "No manifest found"}

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        restore_targets = targets or manifest.get("includes", [])
        restored = []

        for target in restore_targets:
            source = backup_path / target
            dest = self._resolve_source(target)
            if not source.exists() or not dest:
                continue
            try:
                if source.is_dir():
                    if dest.exists():
                        shutil.rmtree(dest)
                    shutil.copytree(source, dest)
                else:
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy2(source, dest)
                restored.append(target)
            except Exception as e:
                logger.warning("Failed to restore %s: %s", target, e)

        return {"success": True, "restored": restored}

    def list_backups(self) -> List[BackupManifest]:
        """List available backups."""
        if not self._backup_dir.exists():
            return []
        results = []
        for d in sorted(self._backup_dir.iterdir(), key=lambda p: -p.stat().st_mtime):
            if not d.is_dir():
                continue
            manifest_path = d / "manifest.json"
            if manifest_path.exists():
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    results.append(BackupManifest(path=str(d), **data))
                except Exception as exc:
                    logger.debug("list_backups: suppressed %s", exc)
        return results

    def delete(self, backup_id: str) -> bool:
        """Delete a backup."""
        path = self._backup_dir / backup_id
        if path.exists():
            shutil.rmtree(path)
            return True
        return False

    def cleanup(self, keep_last: int = 5) -> int:
        """Remove old backups, keeping the most recent."""
        backups = self.list_backups()
        if len(backups) <= keep_last:
            return 0
        removed = 0
        for backup in backups[keep_last:]:
            if self.delete(backup.id):
                removed += 1
        return removed

    def _resolve_source(self, target: str) -> Optional[Path]:
        """Resolve a backup target to its source path."""
        mapping = {
            "config": self._home / "config.json",
            "memory": self._home / "memory",
            "skills": self._home / "skills",
            "sessions": self._home / "sessions",
            "checkpoints": self._home / "checkpoints",
        }
        return mapping.get(target)

    def _dir_size(self, path: Path) -> int:
        """Calculate total size of a directory."""
        total = 0
        for f in path.rglob("*"):
            if f.is_file():
                total += f.stat().st_size
        return total
