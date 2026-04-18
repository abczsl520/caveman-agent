"""Engine feature flags — config-driven + runtime toggle.

Controls which engines (shield, nudge, ripple, lint, recall) are active.
Config sets defaults; runtime API allows dynamic enable/disable.
"""
from __future__ import annotations

import logging
from caveman.errors import CavemanError

logger = logging.getLogger(__name__)


class EngineError(CavemanError):
    """Engine-related error (unknown engine, invalid state)."""
    pass


ENGINES = ("shield", "nudge", "ripple", "lint", "recall", "scheduler", "verification", "reflect")


class EngineFlags:
    """Feature flags for engines. Config-driven + runtime toggle."""

    def __init__(self, config: dict | None = None) -> None:
        self._flags: dict[str, bool] = {}
        engines_cfg = (config or {}).get("engines", {})
        for name in ENGINES:
            engine_cfg = engines_cfg.get(name, {})
            self._flags[name] = engine_cfg.get("enabled", True)

    def _validate(self, engine: str) -> None:
        if engine not in ENGINES:
            raise EngineError(
                f"Unknown engine: {engine!r}",
                context={"engine": engine, "valid": list(ENGINES)},
            )

    def is_enabled(self, engine: str) -> bool:
        """Check if an engine is enabled."""
        self._validate(engine)
        return self._flags[engine]

    def enable(self, engine: str) -> None:
        """Enable an engine at runtime."""
        self._validate(engine)
        self._flags[engine] = True
        logger.info("Engine %s enabled", engine)

    def disable(self, engine: str) -> None:
        """Disable an engine at runtime."""
        self._validate(engine)
        self._flags[engine] = False
        logger.info("Engine %s disabled", engine)

    def status(self) -> dict[str, bool]:
        """Return current status of all engines."""
        return dict(self._flags)
