"""Stream primitives — StreamEvent + StreamBuffer.

The streaming implementation lives in AgentLoop.run_stream() (loop.py).
This module only contains the data types to avoid circular imports.
"""
from __future__ import annotations

__all__ = ["StreamEvent", "StreamBuffer"]

import time
from dataclasses import dataclass, field
from typing import Any


@dataclass
class StreamEvent:
    """A single streaming event from the agent loop."""

    type: str  # token, tool_call, tool_result, thinking, done, error
    data: Any = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {"type": self.type, "data": self.data, "ts": self.timestamp}


class StreamBuffer:
    """Collects StreamEvents and accumulates text."""

    def __init__(self) -> None:
        self._events: list[StreamEvent] = []
        self._text = ""

    def add(self, event: StreamEvent) -> None:
        self._events.append(event)
        if event.type == "token":
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
