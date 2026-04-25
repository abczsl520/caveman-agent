"""Agent Runner Depth — streaming output, tool progress, block reply.

Supplements agent_runner.py with streaming support and tool execution
status tracking. Extracted from OpenClaw agent-runner.ts (874 lines).
"""
from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

__all__ = [
    "ToolProgress",
    "StreamEvent",
    "StreamingAgentRunner",
]


logger = logging.getLogger("caveman.gateway.agent_runner_depth")


@dataclass
class ToolProgress:
    """Progress of a tool execution."""
    tool_name: str
    call_id: str = ""
    status: str = "pending"  # pending | running | completed | failed
    started_at: float = 0
    completed_at: float = 0
    result_preview: str = ""
    error: str = ""

    @property
    def duration_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.started_at) * 1000 if self.started_at else 0


@dataclass
class StreamEvent:
    """An event from the agent stream."""
    type: str  # text | tool_start | tool_progress | tool_complete | thinking | result | error
    text: str = ""
    tool_name: str = ""
    tool_id: str = ""
    tool_result: str = ""
    tokens: int = 0
    phase: str = ""


class StreamingAgentRunner:
    """Agent runner with streaming output and tool progress tracking."""

    def __init__(
        self,
        agent_stream_fn: Optional[Callable] = None,
        on_event: Optional[Callable] = None,
        typing_fn: Optional[Callable] = None,
        tool_heartbeat_interval: float = 15.0,
    ):
        self._agent_stream_fn = agent_stream_fn
        self._on_event = on_event
        self._typing_fn = typing_fn
        self._heartbeat_interval = tool_heartbeat_interval
        self._active_tools: Dict[str, ToolProgress] = {}
        self._pending_tool_tasks: Set[asyncio.Task] = set()
        self._cancel_event: Optional[asyncio.Event] = None

    async def run_streaming(
        self,
        session_key: str,
        body: str,
        model: str = "",
        **kwargs,
    ) -> AsyncIterator[StreamEvent]:
        """Run agent with streaming output."""
        self._cancel_event = asyncio.Event()

        if not self._agent_stream_fn:
            yield StreamEvent(type="error", text="No agent stream function configured")
            return

        try:
            stream = self._agent_stream_fn(
                session_key=session_key, body=body, model=model, **kwargs,
            )
            if hasattr(stream, "__aiter__"):
                async for chunk in stream:
                    if self._cancel_event.is_set():
                        yield StreamEvent(type="result", text="Cancelled")
                        return

                    event = self._process_chunk(chunk)
                    if event:
                        yield event
                        if self._on_event:
                            try:
                                result = self._on_event(event)
                                if hasattr(result, "__await__"):
                                    await result
                            except Exception as exc:
                                logger.debug("run_streaming: suppressed %s", exc)
            else:
                # Non-streaming fallback
                result = stream
                if hasattr(result, "__await__"):
                    result = await result
                text = result.get("text", str(result)) if isinstance(result, dict) else str(result)
                yield StreamEvent(type="text", text=text)

        except asyncio.CancelledError:
            yield StreamEvent(type="result", text="Cancelled")
        except Exception as e:
            yield StreamEvent(type="error", text=str(e))

        yield StreamEvent(type="result")

    def _process_chunk(self, chunk: Any) -> Optional[StreamEvent]:
        """Process a raw chunk from the agent stream."""
        if isinstance(chunk, str):
            return StreamEvent(type="text", text=chunk)

        if isinstance(chunk, dict):
            chunk_type = chunk.get("type", "text")

            if chunk_type == "text":
                return StreamEvent(type="text", text=chunk.get("text", ""))

            elif chunk_type == "tool_use":
                tool_name = chunk.get("name", "")
                tool_id = chunk.get("id", "")
                self._active_tools[tool_id] = ToolProgress(
                    tool_name=tool_name,
                    call_id=tool_id,
                    status="running",
                    started_at=time.monotonic(),
                )
                return StreamEvent(
                    type="tool_start",
                    tool_name=tool_name,
                    tool_id=tool_id,
                )

            elif chunk_type == "tool_result":
                tool_id = chunk.get("id", "")
                progress = self._active_tools.get(tool_id)
                if progress:
                    progress.status = "completed"
                    progress.completed_at = time.monotonic()
                    progress.result_preview = str(chunk.get("result", ""))[:500]
                return StreamEvent(
                    type="tool_complete",
                    tool_name=chunk.get("name", ""),
                    tool_id=tool_id,
                    tool_result=str(chunk.get("result", ""))[:1000],
                )

            elif chunk_type == "thinking":
                return StreamEvent(type="thinking", text=chunk.get("text", ""))

            elif chunk_type == "usage":
                return StreamEvent(
                    type="result",
                    tokens=chunk.get("total_tokens", 0),
                )

        return None

    async def cancel(self) -> None:
        """Cancel the current streaming run."""
        if self._cancel_event:
            self._cancel_event.set()
        for task in self._pending_tool_tasks:
            if not task.done():
                task.cancel()

    def get_active_tools(self) -> List[Dict[str, Any]]:
        """Get currently active tool executions."""
        return [
            {
                "name": p.tool_name,
                "status": p.status,
                "duration_ms": round(p.duration_ms, 1),
            }
            for p in self._active_tools.values()
            if p.status == "running"
        ]

    async def tool_heartbeat(self, send_fn: Callable) -> None:
        """Send periodic heartbeat for long-running tools."""
        while True:
            await asyncio.sleep(self._heartbeat_interval)
            active = self.get_active_tools()
            if active:
                names = ", ".join(t["name"] for t in active)
                durations = ", ".join(f"{t['duration_ms']/1000:.0f}s" for t in active)
                try:
                    result = send_fn(f"⏳ Running {names} ({durations})...")
                    if hasattr(result, "__await__"):
                        await result
                except Exception as exc:
                    logger.debug("tool_heartbeat: suppressed %s", exc)
