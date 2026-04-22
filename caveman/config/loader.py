"""Config loader — load and validate Caveman configuration."""
from __future__ import annotations
import logging
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from caveman.paths import CONFIG_PATH

__all__ = [
    "DEFAULT_CONFIG_PATH",
    "BUNDLED_DEFAULT",
    "invalidate_config_cache",
    "load_config",
]


logger = logging.getLogger(__name__)

DEFAULT_CONFIG_PATH = CONFIG_PATH
BUNDLED_DEFAULT = Path(__file__).parent / "default.yaml"

# ── Config Cache ──
# Avoids re-reading + re-parsing YAML on every load_config() call.
# Invalidated when file mtime changes.
_cache: dict[str, Any] = {}  # key → {"mtime": float, "config": dict}


def _cache_key(user_path: Path) -> str:
    return str(user_path.resolve())


def _get_mtime(path: Path) -> float:
    try:
        return path.stat().st_mtime
    except OSError:
        return 0.0


def invalidate_config_cache() -> None:
    """Clear the config cache. Useful after config file edits."""
    _cache.clear()
    logger.debug("Config cache invalidated")


def _resolve_env_vars(obj: Any) -> Any:
    """Recursively resolve ${ENV_VAR} references."""
    if isinstance(obj, str):
        return re.sub(r'\$\{([^}]+)\}', lambda m: os.environ.get(m.group(1), ""), obj)
    elif isinstance(obj, dict):
        return {k: _resolve_env_vars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_resolve_env_vars(i) for i in obj]
    return obj


def _deep_merge(base: dict, override: dict) -> dict:
    result = base.copy()
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = _deep_merge(result[k], v)
        else:
            result[k] = v
    return result


def load_config(config_path: Path | str | None = None, validate: bool = True) -> dict:
    """Load config: bundled defaults → user config → env vars → validate.

    Results are cached by file path + mtime. Subsequent calls with the same
    config file return the cached dict unless the file has been modified.
    """
    if yaml is None:
        raise ImportError("pyyaml required")

    user_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    key = _cache_key(user_path)

    # Check cache: both bundled default and user config mtimes
    bundled_mtime = _get_mtime(BUNDLED_DEFAULT)
    user_mtime = _get_mtime(user_path)
    combined_mtime = bundled_mtime + user_mtime

    cached = _cache.get(key)
    if cached and cached["mtime"] == combined_mtime:
        return cached["config"]

    # Cache miss — load from disk
    config = {}
    if BUNDLED_DEFAULT.exists():
        with open(BUNDLED_DEFAULT, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    if user_path.exists():
        with open(user_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    resolved = _resolve_env_vars(config)

    if validate:
        from caveman.config.validator import validate_config
        validate_config(resolved, strict=False)

    _cache[key] = {"mtime": combined_mtime, "config": resolved}
    logger.debug("Config loaded and cached: %s", key)
    return resolved
