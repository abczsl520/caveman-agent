"""Structured result types — consistent return shapes across the entire framework.

Problem: 21+ places return ad-hoc {"error": ...} dicts with inconsistent keys.
Solution: ToolResult/Result as the universal result envelope.

Every function that can fail returns a Result. Callers check .ok instead of
guessing which key to look for.

Usage:
    from caveman.result import ToolResult, Ok, Err

    def my_function() -> ToolResult:
        if something_wrong:
            return Err("what went wrong")
        return Ok({"data": "here"})

    r = my_function()
    if r.ok:
        print(r.data)  # {"data": "here"}
    else:
        print(r.error)  # "what went wrong"
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class ToolResult:
    """Universal result envelope for tool/function returns.

    Attributes:
        ok: True if operation succeeded
        data: Result payload (on success)
        error: Error message (on failure)
        metadata: Optional extra info (timing, source, etc.)
    """
    ok: bool
    data: dict[str, Any] = field(default_factory=dict)
    error: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to a consistent dict shape."""
        result: dict[str, Any] = {"ok": self.ok}
        if self.ok:
            result.update(self.data)
        else:
            result["error"] = self.error
        if self.metadata:
            result["_meta"] = self.metadata
        return result

    def to_content(self) -> str:
        """Serialize to string for LLM tool_result content."""
        import json
        return json.dumps(self.to_dict(), ensure_ascii=False)

    @property
    def is_error(self) -> bool:
        return not self.ok


def Ok(data: dict[str, Any] | None = None, **kwargs) -> ToolResult:
    """Create a success result."""
    return ToolResult(ok=True, data=data or kwargs)


def Err(error: str, **metadata) -> ToolResult:
    """Create an error result."""
    return ToolResult(ok=False, error=error, metadata=metadata)
