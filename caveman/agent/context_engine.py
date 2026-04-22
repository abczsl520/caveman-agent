"""Context Engine — pluggable context management interface.

Defines the ContextEngine protocol and provides a default implementation
that combines compaction, pruning, and assembly. Extracted from
OpenClaw src/context-engine/ (1896 lines).
"""
from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

__all__ = [
    "AssembleResult",
    "CompactResult",
    "IngestResult",
    "BootstrapResult",
    "EngineInfo",
    "ContextEngine",
    "DefaultContextEngine",
    "register_engine",
    "get_engine",
    "list_engines",
]


logger = logging.getLogger("caveman.agent.context_engine")


# ── Result Types ──

@dataclass
class AssembleResult:
    """Result of context assembly."""
    messages: List[Dict[str, Any]]
    estimated_tokens: int = 0
    system_prompt_addition: str = ""


@dataclass
class CompactResult:
    """Result of context compaction."""
    ok: bool = True
    compacted: bool = False
    reason: str = ""
    summary: str = ""
    tokens_before: int = 0
    tokens_after: int = 0


@dataclass
class IngestResult:
    """Result of message ingestion."""
    ingested: bool = True


@dataclass
class BootstrapResult:
    """Result of engine bootstrap."""
    bootstrapped: bool = True
    imported_messages: int = 0
    reason: str = ""


@dataclass
class EngineInfo:
    """Engine metadata."""
    id: str
    name: str
    version: str = "1.0"
    owns_compaction: bool = False


# ── Context Engine Protocol ──

class ContextEngine(ABC):
    """Pluggable context management engine.

    Defines the lifecycle:
    1. bootstrap() — initialize for a session
    2. ingest() — add messages to the store
    3. assemble() — build context under token budget
    4. compact() — reduce context size
    5. after_turn() — post-turn lifecycle
    """

    @property
    @abstractmethod
    def info(self) -> EngineInfo: ...

    def bootstrap(
        self, session_id: str, session_file: str = "", **kwargs,
    ) -> BootstrapResult:
        return BootstrapResult()

    @abstractmethod
    def ingest(
        self, session_id: str, message: Dict[str, Any], **kwargs,
    ) -> IngestResult: ...

    @abstractmethod
    def assemble(
        self, session_id: str, messages: List[Dict[str, Any]],
        token_budget: int = 128000, **kwargs,
    ) -> AssembleResult: ...

    @abstractmethod
    def compact(
        self, session_id: str, session_file: str = "",
        token_budget: int = 128000, force: bool = False, **kwargs,
    ) -> CompactResult: ...

    def after_turn(
        self, session_id: str, messages: List[Dict[str, Any]], **kwargs,
    ) -> None:
        pass

    def dispose(self) -> None:
        pass


# ── Default Engine (Compressor-based) ──

class DefaultContextEngine(ContextEngine):
    """Default context engine using the built-in compressor."""

    def __init__(self, model: str = "", threshold_percent: float = 0.50):
        from caveman.agent.context_compressor import ContextCompressor
        self._compressor = ContextCompressor(
            model=model, threshold_percent=threshold_percent,
        )
        self._sessions: Dict[str, List[Dict[str, Any]]] = {}

    @property
    def info(self) -> EngineInfo:
        return EngineInfo(
            id="default",
            name="DefaultContextEngine",
            version="1.0",
            owns_compaction=True,
        )

    def ingest(
        self, session_id: str, message: Dict[str, Any], **kwargs,
    ) -> IngestResult:
        if session_id not in self._sessions:
            self._sessions[session_id] = []
        self._sessions[session_id].append(message)
        return IngestResult(ingested=True)

    def assemble(
        self, session_id: str, messages: List[Dict[str, Any]],
        token_budget: int = 128000, **kwargs,
    ) -> AssembleResult:
        from caveman.agent.context_compressor import estimate_tokens_rough

        # If under budget, return as-is
        tokens = estimate_tokens_rough(messages)
        if tokens <= token_budget:
            return AssembleResult(messages=messages, estimated_tokens=tokens)

        # Need compaction
        result = self._compressor.compress(messages)
        return AssembleResult(
            messages=result.messages,
            estimated_tokens=result.compacted_tokens,
        )

    def compact(
        self, session_id: str, session_file: str = "",
        token_budget: int = 128000, force: bool = False, **kwargs,
    ) -> CompactResult:
        messages = self._sessions.get(session_id, [])
        if not messages:
            return CompactResult(ok=True, compacted=False, reason="no_messages")

        if not force and not self._compressor.should_compress(messages):
            return CompactResult(ok=True, compacted=False, reason="below_threshold")

        result = self._compressor.compress(messages)
        self._sessions[session_id] = result.messages

        return CompactResult(
            ok=True,
            compacted=True,
            summary=result.summary,
            tokens_before=result.original_tokens,
            tokens_after=result.compacted_tokens,
        )


# ── Engine Registry ──

_engines: Dict[str, ContextEngine] = {}


def register_engine(engine: ContextEngine) -> None:
    """Register a context engine."""
    _engines[engine.info.id] = engine


def get_engine(engine_id: str = "default") -> Optional[ContextEngine]:
    """Get a registered context engine."""
    return _engines.get(engine_id)


def list_engines() -> List[EngineInfo]:
    """List registered engines."""
    return [e.info for e in _engines.values()]
