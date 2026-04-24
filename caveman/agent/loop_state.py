"""LoopState — single source of truth for serializable session state.

Adding a field here automatically propagates to snapshot/restore/reset.
"""
from __future__ import annotations
from dataclasses import dataclass, fields, asdict


@dataclass
class LoopState:
    turn_number: int = 0
    turn_count: int = 0
    tool_call_count: int = 0
    iteration_count: int = 0
    surface: str = "cli"

    def snapshot(self) -> dict:
        return asdict(self)

    @classmethod
    def from_snapshot(cls, data: dict) -> LoopState:
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in known})
