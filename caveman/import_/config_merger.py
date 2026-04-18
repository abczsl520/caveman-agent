"""Config merger — extract useful settings from external configs."""
from __future__ import annotations

import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConfigMerger:
    """Merge external configs into Caveman config without overwriting user values."""

    def __init__(self, caveman_home: Path) -> None:
        self.caveman_home = caveman_home
        self.config_path = caveman_home / "config.yaml"

    def merge_openclaw_json(self, raw: str) -> dict:
        """Extract providers/models from openclaw.json content."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid openclaw.json")
            return {}

        extracted: dict = {}
        providers = data.get("providers", data.get("models", {}))
        if isinstance(providers, dict):
            extracted["providers"] = {}
            for name, cfg in providers.items():
                if isinstance(cfg, dict):
                    clean = {k: v for k, v in cfg.items()
                             if k in ("model", "base_url", "max_tokens")}
                    if clean:
                        extracted["providers"][name] = clean

        self._save_extracted(extracted, "openclaw")
        return extracted

    def merge_hermes_yaml(self, raw: str) -> dict:
        """Extract model/providers from Hermes config.yaml content."""
        try:
            import yaml
            data = yaml.safe_load(raw) or {}
        except Exception:
            logger.warning("Invalid hermes config.yaml")
            return {}

        extracted: dict = {}
        if "model" in data:
            extracted["default_model"] = data["model"]
        providers = data.get("providers", {})
        if isinstance(providers, dict):
            extracted["providers"] = {}
            for name, cfg in providers.items():
                if isinstance(cfg, dict):
                    clean = {k: v for k, v in cfg.items()
                             if k in ("model", "base_url", "max_tokens")}
                    if clean:
                        extracted["providers"][name] = clean

        self._save_extracted(extracted, "hermes")
        return extracted

    def merge_claude_settings(self, raw: str) -> dict:
        """Extract model/env from Claude Code settings.json."""
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            logger.warning("Invalid Claude Code settings.json")
            return {}

        extracted: dict = {}
        if "model" in data:
            extracted["default_model"] = data["model"]
        if "env" in data and isinstance(data["env"], dict):
            extracted["env"] = data["env"]

        self._save_extracted(extracted, "claude-code")
        return extracted

    def _save_extracted(self, extracted: dict, source: str) -> None:
        """Save extracted config as a reference file (not auto-applied)."""
        if not extracted:
            return
        ref_dir = self.caveman_home / "imported-configs"
        ref_dir.mkdir(parents=True, exist_ok=True)
        ref_path = ref_dir / f"{source}.json"
        ref_path.write_text(
            json.dumps(extracted, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        logger.info("Saved extracted config: %s", ref_path)
