"""Compaction Safeguard — delay compression during critical phases.

Inspired by OpenClaw compaction-safeguard.ts (MIT, Peter Steinberger).
Prevents compaction from interrupting critical operations like:
- Active tool execution
- Multi-step code generation
- Test runs in progress
- File write sequences

Also provides quality gates for compaction summaries.
"""
from __future__ import annotations

import logging
import re
import time
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

logger = logging.getLogger(__name__)

# Required sections in compaction summary (from OpenClaw)
REQUIRED_SECTIONS = [
    "## Goal",
    "## Key Decisions",
    "## Remaining Work",
    "## Constraints/Rules",
    "## Pending User Asks",
]

# Identifier patterns to preserve during compaction
_IDENTIFIER_PATTERNS = [
    r"[a-f0-9]{7,40}",                    # Git hashes
    r"https?://\S+",                       # URLs
    r"/[\w/.-]+\.\w+",                     # File paths
    r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b",  # IP addresses
    r"\b\d{4,5}\b",                        # Port numbers
    r"\b[A-Z][A-Z0-9_]{2,}\b",            # Constants
    r"v\d+\.\d+(?:\.\d+)?",               # Version numbers
]

_ID_RE = re.compile("|".join(f"({p})" for p in _IDENTIFIER_PATTERNS))


class SafeguardPhase(Enum):
    """Current agent phase — determines if compaction is safe."""
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_EXECUTING = "tool_executing"
    CODE_GENERATING = "code_generating"
    TESTING = "testing"
    FILE_WRITING = "file_writing"
    COMPACTING = "compacting"


# Phases where compaction should be delayed
_CRITICAL_PHASES = frozenset({
    SafeguardPhase.TOOL_EXECUTING,
    SafeguardPhase.CODE_GENERATING,
    SafeguardPhase.TESTING,
    SafeguardPhase.FILE_WRITING,
    SafeguardPhase.COMPACTING,
})


@dataclass
class SafeguardState:
    """Tracks current phase and compaction readiness."""
    phase: SafeguardPhase = SafeguardPhase.IDLE
    phase_started_at: float = 0.0
    pending_compaction: bool = False
    last_compaction_at: float = 0.0
    compaction_count: int = 0
    # Max time to wait for critical phase to end before forcing compaction
    max_wait_seconds: float = 60.0

    @property
    def is_critical(self) -> bool:
        return self.phase in _CRITICAL_PHASES

    @property
    def phase_duration(self) -> float:
        if self.phase_started_at <= 0:
            return 0.0
        return time.time() - self.phase_started_at

    @property
    def should_force(self) -> bool:
        """Force compaction if critical phase has been running too long."""
        return self.is_critical and self.phase_duration > self.max_wait_seconds


class CompactionSafeguard:
    """Gate compaction based on agent phase."""

    def __init__(self, max_wait: float = 60.0) -> None:
        self._state = SafeguardState(max_wait_seconds=max_wait)

    @property
    def state(self) -> SafeguardState:
        return self._state

    def enter_phase(self, phase: SafeguardPhase) -> None:
        """Signal that the agent entered a new phase."""
        self._state.phase = phase
        self._state.phase_started_at = time.time()

        # If compaction was pending and we're now idle, it can proceed
        if phase == SafeguardPhase.IDLE and self._state.pending_compaction:
            logger.info("Critical phase ended, pending compaction can proceed")

    def request_compaction(self) -> bool:
        """Request compaction. Returns True if safe to proceed now."""
        if not self._state.is_critical:
            return True

        if self._state.should_force:
            logger.warning(
                "Forcing compaction after %.0fs in %s phase",
                self._state.phase_duration, self._state.phase.value,
            )
            return True

        # Defer compaction
        self._state.pending_compaction = True
        logger.info(
            "Deferring compaction: agent in %s phase (%.0fs)",
            self._state.phase.value, self._state.phase_duration,
        )
        return False

    def compaction_completed(self) -> None:
        """Signal that compaction finished."""
        self._state.pending_compaction = False
        self._state.last_compaction_at = time.time()
        self._state.compaction_count += 1
        self._state.phase = SafeguardPhase.IDLE

    def has_pending(self) -> bool:
        """Check if there's a deferred compaction waiting."""
        return self._state.pending_compaction


def extract_identifiers(text: str, max_ids: int = 50) -> list[str]:
    """Extract opaque identifiers from text for preservation."""
    seen: set[str] = set()
    result: list[str] = []
    for match in _ID_RE.finditer(text):
        value = match.group(0)
        if value not in seen and len(value) > 3:
            seen.add(value)
            result.append(value)
            if len(result) >= max_ids:
                break
    return result


def audit_summary(summary: str) -> tuple[bool, list[str]]:
    """Check if a compaction summary has all required sections.

    Returns (passed, missing_sections).
    """
    missing = []
    for section in REQUIRED_SECTIONS:
        if section not in summary:
            missing.append(section)
    return len(missing) == 0, missing


def build_compaction_instructions(
    custom: str = "",
    preserve_ids: bool = True,
) -> str:
    """Build instructions for the compaction LLM call."""
    parts = [
        "Produce a compact, factual summary with these exact section headings:",
        *REQUIRED_SECTIONS,
        "## Exact identifiers",
        "",
        "For ## Exact identifiers, preserve literal values exactly as seen "
        "(IDs, URLs, file paths, ports, hashes, dates, times).",
        "Do not omit unresolved asks from the user.",
    ]

    if custom:
        parts.append(f"\nAdditional context:\n{custom}")

    return "\n".join(parts)
