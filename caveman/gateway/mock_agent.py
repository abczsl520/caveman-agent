"""Mock agent and components for gateway slash command handling.

Provides all attributes accessed by command handlers so slash commands
return meaningful data instead of dummy values.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)


class GatewayMockAgent:
    """Gateway-aware agent stub that reads real config and state."""

    def __init__(self):
        self._cfg = {}
        try:
            from caveman.config.loader import load_config
            self._cfg = load_config()
        except Exception as e:
            logger.debug("Suppressed in __init__: %s", e)

        self.model = self._cfg.get("agent", {}).get("default_model", "unknown")
        self.max_iterations = self._cfg.get("agent", {}).get("max_iterations", 50)
        self.reasoning_level = "medium"
        self.fast_mode = False
        self._tool_call_count = 0
        self.conversation = []

        self.tool_registry = _MockToolRegistry()
        self.memory_manager = _MockMemoryManager()
        self.skill_manager = _MockSkillManager()
        self.engine_flags = _MockEngineFlags(self._cfg.get("engines", {}))
        self._shield = _MockShield()
        self._reflect = _MockReflect()
        self.provider = _MockProvider(self._cfg)
        self.bus = None

    async def retry(self):
        pass

    async def compress_context(self, topic=None):
        pass

    async def drain_background(self, timeout=3.0):
        pass


class _MockToolRegistry:
    def get_schemas(self):
        try:
            from caveman.tools.registry import ToolRegistry
            reg = ToolRegistry()
            reg.register_builtins()
            return reg.get_schemas()
        except Exception:
            return []


class _MockMemoryManager:
    @property
    def total_count(self):
        try:
            from caveman.memory.sqlite_store import SQLiteMemoryStore
            from caveman.paths import CAVEMAN_HOME
            store = SQLiteMemoryStore(CAVEMAN_HOME / "memory.db")
            return store.count()
        except Exception:
            return 0


class _MockSkillManager:
    def list_all(self):
        try:
            from caveman.skills.manager import SkillManager
            sm = SkillManager()
            return sm.list_all()
        except Exception:
            return []


class _MockEngineFlags:
    def __init__(self, engines_cfg: dict):
        self._engines = engines_cfg

    def is_enabled(self, name: str) -> bool:
        eng = self._engines.get(name, {})
        return eng.get("enabled", True) if isinstance(eng, dict) else True

    def enable(self, name: str):
        self._engines.setdefault(name, {})["enabled"] = True

    def disable(self, name: str):
        self._engines.setdefault(name, {})["enabled"] = False


class _MockShield:
    class essence:
        turn_count = 0
        task = None
        decisions = []
        progress = []
        open_todos = []
        key_data = {}
        session_id = "gateway"


class _MockReflect:
    reflections = []


class _MockProvider:
    def __init__(self, cfg: dict):
        self.model_name = cfg.get("agent", {}).get("default_model", "unknown")
        self.rate_limit_state = None
