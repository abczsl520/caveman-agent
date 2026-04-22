"""Hot config reload — watch config file for changes and notify listeners.

Uses file mtime polling (no external deps like watchdog).
Listeners register callbacks that receive the new config dict.
"""
from __future__ import annotations
import asyncio
import logging
from pathlib import Path
from typing import Any, Callable, Coroutine

from caveman.config.loader import load_config, invalidate_config_cache, DEFAULT_CONFIG_PATH

logger = logging.getLogger(__name__)

POLL_INTERVAL = 5.0  # Check every 5 seconds

ConfigCallback = Callable[[dict[str, Any]], Coroutine[Any, Any, None]]


class ConfigWatcher:
    """Watch config file for changes and notify listeners."""

    def __init__(self, config_path: Path | str | None = None):
        self._path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
        self._callbacks: list[ConfigCallback] = []
        self._last_mtime: float = 0
        self._last_hash: str = ""
        self._running = False
        self._task: asyncio.Task | None = None

    def on_change(self, callback: ConfigCallback) -> None:
        """Register a callback for config changes."""
        self._callbacks.append(callback)

    def _get_file_state(self) -> tuple[float, str]:
        """Get mtime and content hash of config file."""
        try:
            stat = self._path.stat()
            mtime = stat.st_mtime
            # Quick hash: mtime + size
            return mtime, f"{mtime}:{stat.st_size}"
        except FileNotFoundError:
            return 0, ""

    async def _poll_loop(self) -> None:
        self._last_mtime, self._last_hash = self._get_file_state()
        while self._running:
            await asyncio.sleep(POLL_INTERVAL)
            mtime, file_hash = self._get_file_state()
            if file_hash != self._last_hash and mtime > self._last_mtime:
                self._last_mtime = mtime
                self._last_hash = file_hash
                logger.info("Config file changed, reloading...")
                try:
                    invalidate_config_cache()
                    new_config = load_config(self._path)
                    for cb in self._callbacks:
                        try:
                            await cb(new_config)
                        except Exception as e:
                            logger.warning("Config change callback failed: %s", e)
                    logger.info("Config reloaded successfully")
                except Exception as e:
                    logger.error("Config reload failed: %s", e)

    def start(self) -> None:
        """Start watching (must be called from async context)."""
        if self._running:
            return
        self._running = True
        self._task = asyncio.get_running_loop().create_task(
            self._poll_loop(), name="config_watcher")

    def stop(self) -> None:
        """Stop watching."""
        self._running = False
        if self._task:
            self._task.cancel()
            self._task = None

    async def force_reload(self) -> dict[str, Any]:
        """Force reload config and notify listeners."""
        invalidate_config_cache()
        new_config = load_config(self._path)
        self._last_mtime, self._last_hash = self._get_file_state()
        for cb in self._callbacks:
            try:
                await cb(new_config)
            except Exception as e:
                logger.warning("Config change callback failed: %s", e)
        return new_config
