"""User-defined event hook system.

Discovers hooks from ~/.caveman/hooks/ directories, each containing:
  - HOOK.yaml  (metadata: name, description, events list)
  - handler.py (Python handler with async def handle(event_type, context))

Integrates with Caveman's EventBus — hooks fire on matching EventBus events.
Errors in hooks are caught and logged but never block the main pipeline.

Events (maps to EventBus EventType + gateway lifecycle):
  - gateway:startup     -- Gateway process starts
  - gateway:shutdown    -- Gateway process stops
  - session:start       -- New session created
  - session:end         -- Session ends
  - agent:start         -- Agent begins processing
  - agent:step          -- Each iteration in the tool loop
  - agent:end           -- Agent finishes processing
  - tool:call           -- Tool invoked
  - tool:result         -- Tool returned
  - memory:store        -- Memory stored
  - memory:recall       -- Memory recalled
  - command:*           -- Any slash command (wildcard)
"""
from __future__ import annotations

import asyncio
import importlib.util
import logging
from pathlib import Path
from typing import Any, Callable

from caveman.paths import CAVEMAN_HOME

logger = logging.getLogger(__name__)

HOOKS_DIR = CAVEMAN_HOME / "hooks"

# Map EventBus event types to hook event names
_EVENTBUS_TO_HOOK = {
    "loop.start": "agent:start",
    "loop.end": "agent:end",
    "iteration.start": "agent:step",
    "tool.call": "tool:call",
    "tool.result": "tool:result",
    "memory.store": "memory:store",
    "memory.recall": "memory:recall",
}


class HookRegistry:
    """Discovers, loads, and fires user-defined event hooks.

    Usage:
        registry = HookRegistry()
        registry.discover_and_load()
        await registry.emit("agent:start", {"session_id": "..."})
    """

    def __init__(self, hooks_dir: Path | None = None) -> None:
        self._hooks_dir = hooks_dir or HOOKS_DIR
        self._handlers: dict[str, list[Callable]] = {}
        self._loaded_hooks: list[dict[str, Any]] = []

    @property
    def loaded_hooks(self) -> list[dict[str, Any]]:
        """Metadata about all loaded hooks."""
        return list(self._loaded_hooks)

    def discover_and_load(self) -> int:
        """Scan hooks directory and load all valid hooks.

        Returns number of hooks loaded.
        """
        if not self._hooks_dir.exists():
            return 0

        count = 0
        for hook_dir in sorted(self._hooks_dir.iterdir()):
            if not hook_dir.is_dir():
                continue

            manifest_path = hook_dir / "HOOK.yaml"
            handler_path = hook_dir / "handler.py"

            if not manifest_path.exists() or not handler_path.exists():
                continue

            try:
                manifest = self._load_manifest(manifest_path)
                if manifest is None:
                    continue

                hook_name = manifest.get("name", hook_dir.name)
                events = manifest.get("events", [])
                if not events:
                    logger.warning("Hook '%s': no events declared, skipping", hook_name)
                    continue

                handle_fn = self._load_handler(hook_name, handler_path)
                if handle_fn is None:
                    continue

                for event in events:
                    self._handlers.setdefault(event, []).append(handle_fn)

                self._loaded_hooks.append({
                    "name": hook_name,
                    "description": manifest.get("description", ""),
                    "events": events,
                    "path": str(hook_dir),
                })
                count += 1
                logger.info("Loaded hook '%s' for events: %s", hook_name, events)

            except Exception as e:
                logger.error("Error loading hook %s: %s", hook_dir.name, e)

        return count

    async def emit(self, event_type: str, context: dict[str, Any] | None = None) -> None:
        """Fire all handlers registered for an event.

        Supports wildcard: handlers for "command:*" fire for any "command:...".
        """
        if context is None:
            context = {}

        handlers = list(self._handlers.get(event_type, []))

        # Wildcard matching
        if ":" in event_type:
            base = event_type.split(":")[0]
            handlers.extend(self._handlers.get(f"{base}:*", []))

        for fn in handlers:
            try:
                result = fn(event_type, context)
                if asyncio.iscoroutine(result):
                    await result
            except Exception as e:
                logger.error("Hook handler error for '%s': %s", event_type, e)

    def create_eventbus_bridge(self) -> Callable:
        """Return an EventBus handler that bridges events to hooks.

        Usage:
            bus.on_all(registry.create_eventbus_bridge())
        """
        async def _bridge(event) -> None:
            hook_event = _EVENTBUS_TO_HOOK.get(event.type)
            if hook_event:
                await self.emit(hook_event, event.data)

        return _bridge

    @staticmethod
    def _load_manifest(path: Path) -> dict[str, Any] | None:
        """Load and validate a HOOK.yaml manifest."""
        try:
            import yaml
            manifest = yaml.safe_load(path.read_text(encoding="utf-8"))
        except ImportError:
            # Fallback: simple key: value parsing for yaml-less envs
            manifest = {}
            for line in path.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if ":" in line and not line.startswith("#"):
                    k, v = line.split(":", 1)
                    k, v = k.strip(), v.strip()
                    if k == "events":
                        continue  # handled below
                    manifest[k] = v
            # Parse events as list items
            events = []
            in_events = False
            for line in path.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped.startswith("events:"):
                    in_events = True
                    continue
                if in_events:
                    if stripped.startswith("- "):
                        events.append(stripped[2:].strip())
                    elif stripped and not stripped.startswith("#"):
                        in_events = False
            if events:
                manifest["events"] = events
        except Exception as e:
            logger.error("Failed to parse %s: %s", path, e)
            return None

        if not manifest or not isinstance(manifest, dict):
            logger.warning("Invalid HOOK.yaml: %s", path)
            return None
        return manifest

    @staticmethod
    def _load_handler(hook_name: str, path: Path) -> Callable | None:
        """Dynamically load a handler.py module and extract handle()."""
        try:
            spec = importlib.util.spec_from_file_location(
                f"caveman_hook_{hook_name}", path
            )
            if spec is None or spec.loader is None:
                logger.warning("Hook '%s': could not load handler.py", hook_name)
                return None

            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)

            handle_fn = getattr(module, "handle", None)
            if handle_fn is None:
                logger.warning("Hook '%s': no 'handle' function found", hook_name)
                return None

            return handle_fn
        except Exception as e:
            logger.error("Hook '%s' handler load error: %s", hook_name, e)
            return None
