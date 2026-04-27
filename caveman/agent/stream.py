"""Stream primitives — StreamEvent + StreamBuffer.

The streaming implementation lives in AgentLoop.run_stream() (loop.py).
This module only contains the data types to avoid circular imports.
"""
from __future__ import annotations

__all__ = [
    "STREAM_TOKEN",
    "STREAM_TOOL_CALL",
    "STREAM_TOOL_RESULT",
    "STREAM_THINKING",
    "STREAM_RESULT",
    "STREAM_ERROR",
    "StreamEvent",
    "StreamBuffer",
    "is_result_event_type",
]

import time
from dataclasses import dataclass, field
from typing import Any

STREAM_TOKEN = "token"
STREAM_TOOL_CALL = "tool_call"
STREAM_TOOL_RESULT = "tool_result"
STREAM_THINKING = "thinking"
STREAM_RESULT = "result"
STREAM_ERROR = "error"
_FORBIDDEN_LEGACY_RESULT_EVENT = "do" "ne"


def is_result_event_type(event_type: str) -> bool:
    """Return True only for the canonical result event.

    The old internal stream event name ``done`` is deliberately *not* accepted
    anymore. Treating an agent turn boundary as a "done" event leaked through
    gateway/API consumers as a false completion signal.
    """
    return event_type == STREAM_RESULT


@dataclass
class StreamEvent:
    """A single streaming event from the agent loop."""

    type: str  # token, tool_call, tool_result, thinking, result, error
    data: Any = ""
    timestamp: float = field(default_factory=time.time)

    def __post_init__(self) -> None:
        if self.type == _FORBIDDEN_LEGACY_RESULT_EVENT:
            raise ValueError(
                "legacy stream event type 'done' is forbidden; use 'result' "
                "for internal turn results and keep completion semantics explicit"
            )

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "ts": self.timestamp}


class StreamBuffer:
    """Collects StreamEvents and accumulates text."""

    def __init__(self) -> None:
        self._events: list[StreamEvent] = []
        self._text = ""

    def add(self, event: StreamEvent) -> None:
        self._events.append(event)
        if event.type == STREAM_TOKEN:
            self._text += str(event.data)

    @property
    def text(self) -> str:
        return self._text

    @property
    def events(self) -> list[StreamEvent]:
        return list(self._events)

    def __len__(self) -> int:
        return len(self._events)

    def clear(self) -> None:
        self._events.clear()
        self._text = ""
