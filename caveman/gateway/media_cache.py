"""Sticker/media cache — avoid re-downloading the same media.

Caches downloaded stickers, images, and other media files by content hash.
Reduces bandwidth and latency for repeated media in chat platforms.
"""
from __future__ import annotations

import hashlib
import logging
from pathlib import Path
from typing import Any

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

_CACHE_DIR = CAVEMAN_HOME / "cache" / "media"
_MAX_CACHE_SIZE_MB = 100
_MAX_AGE_DAYS = 30


class MediaCache:
    """Content-addressed media cache."""

    def __init__(self, cache_dir: Path | None = None, max_size_mb: int = _MAX_CACHE_SIZE_MB) -> None:
        self._dir = cache_dir or _CACHE_DIR
        self._max_bytes = max_size_mb * 1024 * 1024
        self._hits = 0
        self._misses = 0

    def get(self, url: str) -> Path | None:
        """Get cached file path for a URL, or None if not cached."""
        key = self._url_key(url)
        if not self._dir.exists():
            self._misses += 1
            return None
        # Check with and without extension
        for f in self._dir.iterdir():
            if f.is_file() and f.name.startswith(key):
                f.touch()
                self._hits += 1
                return f
        self._misses += 1
        return None

    def put(self, url: str, data: bytes, extension: str = "") -> Path:
        """Store media data in cache. Returns the cached file path."""
        self._dir.mkdir(parents=True, exist_ok=True)
        key = self._url_key(url)
        if extension:
            key = f"{key}{extension}"
        cached = self._dir / key
        cached.write_bytes(data)
        self._evict_if_needed()
        return cached

    def has(self, url: str) -> bool:
        """Check if URL is in cache."""
        return self.get(url) is not None

    def remove(self, url: str) -> bool:
        """Remove a cached entry."""
        cached = self._dir / self._url_key(url)
        if cached.exists():
            cached.unlink(missing_ok=True)
            return True
        return False

    def clear(self) -> int:
        """Clear all cached media. Returns number of files removed."""
        if not self._dir.exists():
            return 0
        count = 0
        for f in self._dir.iterdir():
            if f.is_file():
                f.unlink(missing_ok=True)
                count += 1
        return count

    @property
    def stats(self) -> dict[str, Any]:
        """Cache statistics."""
        total_size = sum(f.stat().st_size for f in self._dir.iterdir() if f.is_file()) if self._dir.exists() else 0
        file_count = sum(1 for f in self._dir.iterdir() if f.is_file()) if self._dir.exists() else 0
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self._hits / max(self._hits + self._misses, 1),
            "files": file_count,
            "size_mb": round(total_size / (1024 * 1024), 2),
        }

    def _evict_if_needed(self) -> None:
        """Evict oldest files if cache exceeds max size."""
        if not self._dir.exists():
            return

        files = sorted(
            (f for f in self._dir.iterdir() if f.is_file()),
            key=lambda f: f.stat().st_mtime,
        )

        total = sum(f.stat().st_size for f in files)
        while total > self._max_bytes and files:
            oldest = files.pop(0)
            total -= oldest.stat().st_size
            oldest.unlink(missing_ok=True)
            logger.debug("Cache evicted: %s", oldest.name)

    @staticmethod
    def _url_key(url: str) -> str:
        """Generate a cache key from URL."""
        return hashlib.sha256(url.encode()).hexdigest()[:24]
