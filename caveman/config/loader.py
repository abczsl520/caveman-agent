"""Config loader — load and validate Caveman configuration."""
from __future__ import annotations
import os
import re
from pathlib import Path
from typing import Any

try:
    import yaml
except ImportError:
    yaml = None

from caveman.paths import CONFIG_PATH

DEFAULT_CONFIG_PATH = CONFIG_PATH
BUNDLED_DEFAULT = Path(__file__).parent / "default.yaml"


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
    """Load config: bundled defaults → user config → env vars → validate."""
    if yaml is None:
        raise ImportError("pyyaml required")
    config = {}
    if BUNDLED_DEFAULT.exists():
        with open(BUNDLED_DEFAULT, encoding="utf-8") as f:
            config = yaml.safe_load(f) or {}
    user_path = Path(config_path) if config_path else DEFAULT_CONFIG_PATH
    if user_path.exists():
        with open(user_path, encoding="utf-8") as f:
            user_config = yaml.safe_load(f) or {}
        config = _deep_merge(config, user_config)
    resolved = _resolve_env_vars(config)

    if validate:
        from caveman.config.validator import validate_config
        validate_config(resolved, strict=False)  # Log warnings, don't crash

    return resolved
