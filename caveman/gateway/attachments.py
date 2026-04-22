"""Attachment Handler — image, file, and media processing.

Handles attachment uploads, downloads, format conversion,
and size validation for messaging platforms.
"""
from __future__ import annotations

import hashlib
import logging
import mimetypes
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "PLATFORM_SIZE_LIMITS",
    "ALLOWED_IMAGE_TYPES",
    "ALLOWED_AUDIO_TYPES",
    "ALLOWED_VIDEO_TYPES",
    "ALLOWED_DOC_TYPES",
    "Attachment",
    "AttachmentHandler",
]


logger = logging.getLogger("caveman.gateway.attachments")

_ATTACHMENTS_DIR = Path.home() / ".caveman" / "attachments"

# Platform attachment limits (bytes)
PLATFORM_SIZE_LIMITS = {
    "discord": 25 * 1024 * 1024,  # 25MB (free), 50MB (nitro)
    "telegram": 50 * 1024 * 1024,  # 50MB
    "whatsapp": 16 * 1024 * 1024,  # 16MB
    "slack": 1024 * 1024 * 1024,  # 1GB
    "signal": 100 * 1024 * 1024,  # 100MB
    "default": 25 * 1024 * 1024,
}

ALLOWED_IMAGE_TYPES = {"image/png", "image/jpeg", "image/gif", "image/webp"}
ALLOWED_AUDIO_TYPES = {"audio/mpeg", "audio/ogg", "audio/wav", "audio/mp4"}
ALLOWED_VIDEO_TYPES = {"video/mp4", "video/webm"}
ALLOWED_DOC_TYPES = {"application/pdf", "text/plain", "text/csv", "application/json"}


@dataclass
class Attachment:
    """A message attachment."""
    filename: str
    mime_type: str = ""
    size: int = 0
    url: str = ""
    local_path: str = ""
    content_hash: str = ""
    downloaded: bool = False

    @property
    def is_image(self) -> bool:
        return self.mime_type in ALLOWED_IMAGE_TYPES

    @property
    def is_audio(self) -> bool:
        return self.mime_type in ALLOWED_AUDIO_TYPES

    @property
    def is_video(self) -> bool:
        return self.mime_type in ALLOWED_VIDEO_TYPES

    @property
    def extension(self) -> str:
        ext = mimetypes.guess_extension(self.mime_type)
        return ext or Path(self.filename).suffix or ""


class AttachmentHandler:
    """Handles attachment processing for messaging platforms."""

    def __init__(self, storage_dir: Optional[Path] = None):
        self._storage_dir = storage_dir or _ATTACHMENTS_DIR
        self._storage_dir.mkdir(parents=True, exist_ok=True)

    def validate(self, attachment: Attachment, platform: str = "default") -> Dict[str, Any]:
        """Validate an attachment for a platform."""
        limit = PLATFORM_SIZE_LIMITS.get(platform, PLATFORM_SIZE_LIMITS["default"])

        if attachment.size > limit:
            return {
                "valid": False,
                "reason": f"File too large: {attachment.size:,} bytes (limit: {limit:,})",
            }

        all_allowed = ALLOWED_IMAGE_TYPES | ALLOWED_AUDIO_TYPES | ALLOWED_VIDEO_TYPES | ALLOWED_DOC_TYPES
        if attachment.mime_type and attachment.mime_type not in all_allowed:
            return {
                "valid": False,
                "reason": f"Unsupported file type: {attachment.mime_type}",
            }

        return {"valid": True}

    def download(self, attachment: Attachment) -> Optional[Path]:
        """Download an attachment from URL to local storage."""
        if not attachment.url:
            return None

        import urllib.request
        try:
            # Generate safe filename
            ext = attachment.extension or ".bin"
            safe_name = hashlib.sha256(attachment.url.encode()).hexdigest()[:16] + ext
            local_path = self._storage_dir / safe_name

            req = urllib.request.Request(attachment.url, headers={
                "User-Agent": "CavemanBot/1.0",
            })
            with urllib.request.urlopen(req, timeout=30) as resp:
                data = resp.read()

            local_path.write_bytes(data)
            attachment.local_path = str(local_path)
            attachment.size = len(data)
            attachment.content_hash = hashlib.sha256(data).hexdigest()[:16]
            attachment.downloaded = True

            # Detect mime type if not set
            if not attachment.mime_type:
                guessed = mimetypes.guess_type(attachment.filename)[0]
                if guessed:
                    attachment.mime_type = guessed

            return local_path

        except Exception as e:
            logger.error("Failed to download attachment: %s", e)
            return None

    def cleanup(self, max_age_hours: int = 24) -> int:
        """Remove old attachments."""
        if not self._storage_dir.exists():
            return 0
        cutoff = time.time() - (max_age_hours * 3600)
        removed = 0
        for f in self._storage_dir.iterdir():
            try:
                if f.is_file() and f.stat().st_mtime < cutoff:
                    f.unlink(missing_ok=True)
                    removed += 1
            except Exception:
                pass  # intentional: Exception suppressed
        return removed

    def list_attachments(self) -> List[Dict[str, Any]]:
        """List stored attachments."""
        if not self._storage_dir.exists():
            return []
        return [
            {
                "filename": f.name,
                "size": f.stat().st_size,
                "modified": f.stat().st_mtime,
            }
            for f in self._storage_dir.iterdir()
            if f.is_file()
        ]
