"""Plugin system — extensible capability loading.

Plugins extend Caveman with:
  - New tools (browser automation, database, API clients)
  - New providers (local models, alternative APIs)
  - New skills (domain-specific expertise)
  - New gateways (Slack, Matrix, custom protocols)

Plugin structure:
  ~/.caveman/plugins/my-plugin/
    plugin.yaml          # metadata + entry point
    __init__.py          # Python module

plugin.yaml:
  name: my-plugin
  version: 1.0.0
  description: A useful plugin
  author: someone
  entry_point: __init__.py
  type: tool | provider | skill | gateway
  requires: []           # pip dependencies
"""
from __future__ import annotations
import importlib.util
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import yaml

logger = logging.getLogger(__name__)


@dataclass
class PluginMeta:
    """Plugin metadata from plugin.yaml."""
    name: str
    version: str = "0.1.0"
    description: str = ""
    author: str = ""
    entry_point: str = "__init__.py"
    plugin_type: str = "tool"  # tool | provider | skill | gateway
    requires: list[str] = field(default_factory=list)
    enabled: bool = True
    path: str = ""

    @classmethod
    def from_yaml(cls, yaml_path: Path) -> "PluginMeta":
        with open(yaml_path, encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
        if not isinstance(data, dict):
            raise ValueError(f"Invalid plugin.yaml (expected mapping, got {type(data).__name__}): {yaml_path}")

        # Validate entry_point: no path traversal
        entry_point = data.get("entry_point", "__init__.py")
        resolved = (yaml_path.parent / entry_point).resolve()
        if not str(resolved).startswith(str(yaml_path.parent.resolve())):
            raise ValueError(f"entry_point escapes plugin directory: {entry_point}")

        # Coerce requires to list
        raw_requires = data.get("requires", [])
        if isinstance(raw_requires, str):
            raw_requires = [raw_requires]
        elif not isinstance(raw_requires, list):
            raw_requires = []

        return cls(
            name=data.get("name", yaml_path.parent.name),
            version=str(data.get("version", "0.1.0")),
            description=data.get("description", ""),
            author=data.get("author", ""),
            entry_point=entry_point,
            plugin_type=data.get("type", "tool"),
            requires=raw_requires,
            enabled=data.get("enabled", True),
            path=str(yaml_path.parent),
        )


class PluginManager:
    """Discover, load, and manage plugins."""

    def __init__(self, plugins_dir: str | None = None):
        from caveman.paths import PLUGINS_DIR
        self.plugins_dir = Path(plugins_dir).expanduser() if plugins_dir else PLUGINS_DIR
        self._plugins: dict[str, PluginMeta] = {}
        self._loaded: dict[str, Any] = {}  # name → module
        self._hooks: dict[str, list[Callable]] = {}

    def discover(self) -> list[PluginMeta]:
        """Discover plugins in the plugins directory."""
        if not self.plugins_dir.exists():
            return []

        found = []
        for plugin_dir in sorted(self.plugins_dir.iterdir()):
            if not plugin_dir.is_dir():
                continue
            yaml_path = plugin_dir / "plugin.yaml"
            if not yaml_path.exists():
                continue

            try:
                meta = PluginMeta.from_yaml(yaml_path)
                self._plugins[meta.name] = meta
                found.append(meta)
                logger.debug("Discovered plugin: %s v%s", meta.name, meta.version)
            except Exception as e:
                logger.warning("Failed to parse %s: %s", yaml_path, e)

        return found

    def load(self, name: str) -> Any:
        """Load a plugin by name."""
        if name in self._loaded:
            return self._loaded[name]

        meta = self._plugins.get(name)
        if not meta:
            raise KeyError(f"Plugin not found: {name}")
        if not meta.enabled:
            raise ValueError(f"Plugin disabled: {name}")

        plugin_path = Path(meta.path) / meta.entry_point
        if not plugin_path.exists():
            raise FileNotFoundError(f"Entry point not found: {plugin_path}")

        # Defense-in-depth: block path traversal even if PluginMeta was built manually
        if not str(plugin_path.resolve()).startswith(str(Path(meta.path).resolve())):
            raise ValueError(f"entry_point escapes plugin directory: {meta.entry_point}")

        # Dynamic import
        sys_key = f"caveman_plugin_{name}"
        spec = importlib.util.spec_from_file_location(sys_key, plugin_path)
        if spec is None or spec.loader is None:
            raise ImportError(f"Unable to create import spec for plugin: {plugin_path}")

        module = importlib.util.module_from_spec(spec)
        try:
            sys.modules[sys_key] = module
            spec.loader.exec_module(module)

            # Call setup() if defined
            if hasattr(module, "setup"):
                module.setup()

            self._loaded[name] = module
            logger.info("Loaded plugin: %s v%s", name, meta.version)
            return module
        except Exception:
            sys.modules.pop(sys_key, None)
            self._loaded.pop(name, None)
            raise

    def load_all(self) -> int:
        """Discover and load all enabled plugins."""
        self.discover()
        count = 0
        for name, meta in self._plugins.items():
            if meta.enabled:
                try:
                    self.load(name)
                    count += 1
                except Exception as e:
                    logger.error("Failed to load plugin %s: %s", name, e)
        return count

    def get(self, name: str) -> Any:
        """Get a loaded plugin module."""
        return self._loaded.get(name)

    def list_all(self) -> list[PluginMeta]:
        """List all discovered plugins."""
        return list(self._plugins.values())

    def list_loaded(self) -> list[str]:
        """List loaded plugin names."""
        return list(self._loaded.keys())

    def unload(self, name: str) -> bool:
        """Unload a plugin."""
        module = self._loaded.pop(name, None)
        if module and hasattr(module, "teardown"):
            try:
                module.teardown()
            except Exception as e:
                logger.warning("Plugin %s teardown error: %s", name, e)

        # Remove from sys.modules
        sys_key = f"caveman_plugin_{name}"
        sys.modules.pop(sys_key, None)
        return module is not None

    # ── Hook system ──

    def register_hook(self, event: str, callback: Callable) -> None:
        """Register a hook callback."""
        self._hooks.setdefault(event, []).append(callback)

    async def emit_hook(self, event: str, **kwargs) -> list[Any]:
        """Emit a hook event, calling all registered callbacks."""
        results = []
        for cb in self._hooks.get(event, []):
            try:
                import asyncio
                if asyncio.iscoroutinefunction(cb):
                    result = await cb(**kwargs)
                else:
                    result = cb(**kwargs)
                results.append(result)
            except Exception as e:
                logger.error("Hook %s callback error: %s", event, e)
        return results
