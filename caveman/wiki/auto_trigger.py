"""Wiki auto-trigger — compile wiki when enough new memories accumulate.

Listens to NUDGE_EXTRACT events and triggers wiki compilation when
the accumulation threshold is reached. This closes the knowledge
crystallization loop:

  Nudge extracts memories → accumulate → Wiki compiles → structured knowledge
  → better system prompt context → better responses → better memories

PRD §5.2 Ring 5: "Wiki is the crystallized knowledge layer."
"""
from __future__ import annotations

import logging
import time
from typing import Any

logger = logging.getLogger(__name__)

__all__ = ["WikiAutoTrigger"]

# Minimum memories accumulated before triggering compilation
_DEFAULT_THRESHOLD = 5
# Minimum seconds between compilations (prevent thrashing)
_DEFAULT_COOLDOWN = 300  # 5 minutes


class WikiAutoTrigger:
    """Accumulates nudge events and triggers wiki compilation at threshold."""

    def __init__(
        self,
        compiler: Any = None,
        memory_manager: Any = None,
        threshold: int = _DEFAULT_THRESHOLD,
        cooldown: float = _DEFAULT_COOLDOWN,
    ) -> None:
        self._compiler = compiler
        self._memory = memory_manager
        self._threshold = threshold
        self._cooldown = cooldown
        self._accumulated = 0
        self._last_compile_ts = 0.0

    @property
    def accumulated(self) -> int:
        return self._accumulated

    def on_nudge_extract(self, count: int = 1) -> bool:
        """Record new nudge extractions. Returns True if compilation triggered."""
        self._accumulated += count
        if self._accumulated >= self._threshold:
            elapsed = time.monotonic() - self._last_compile_ts
            if elapsed >= self._cooldown or self._last_compile_ts == 0:
                return self._trigger_compile()
        return False

    def _trigger_compile(self) -> bool:
        """Run wiki compilation and ingest high-trust memories."""
        if not self._compiler:
            logger.debug("WikiAutoTrigger: no compiler configured")
            return False

        try:
            # Ingest high-trust memories into wiki working tier
            ingested = self._ingest_high_trust_memories()

            # Run compilation (promote + expire + consolidate)
            result = self._compiler.compile()

            self._last_compile_ts = time.monotonic()
            self._accumulated = 0

            logger.info(
                "Wiki auto-compiled: ingested=%d, promoted=%d, expired=%d, total=%d",
                ingested, result.entries_promoted,
                result.entries_expired, result.entries_processed,
            )
            return True
        except Exception as e:
            logger.warning("Wiki auto-compile failed: %s", e)
            return False

    def _ingest_high_trust_memories(self) -> int:
        """Pull high-trust memories from memory store into wiki."""
        if not self._memory:
            return 0

        count = 0
        try:
            # Get memories with high trust that haven't been wiki-ingested
            if hasattr(self._memory, 'all_entries'):
                entries = self._memory.all_entries
                all_mems = entries() if callable(entries) else entries
            elif hasattr(self._memory, '_backend') and hasattr(self._memory._backend, 'all_entries'):
                entries = self._memory._backend.all_entries
                all_mems = entries() if callable(entries) else entries
            else:
                return 0

            for mem in all_mems:
                meta = getattr(mem, "metadata", {}) or {}
                trust = meta.get("trust_score", getattr(mem, "trust_score", 0.5))
                retrieval_count = meta.get("retrieval_count", 0)

                # Only ingest memories that are proven useful:
                # high trust (>0.7) AND actually retrieved (>2 times)
                if trust > 0.7 and retrieval_count > 2:
                    content = getattr(mem, "content", str(mem))
                    source = meta.get("source", "memory")
                    wiki_tag = meta.get("_wiki_ingested", False)
                    if wiki_tag:
                        continue  # already ingested

                    self._compiler.ingest(
                        content=content,
                        source=f"memory:{source}",
                        confidence=trust,
                        tags=["auto-ingested"],
                    )
                    count += 1

        except Exception as e:
            logger.debug("Wiki memory ingestion error: %s", e)

        return count

    def force_compile(self) -> bool:
        """Force immediate compilation regardless of threshold/cooldown."""
        return self._trigger_compile()
