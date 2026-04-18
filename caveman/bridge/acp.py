"""ACP (Agent Communication Protocol) integration.

ACP enables Caveman to:
  1. Be spawned BY OpenClaw as an ACP agent
  2. Spawn OTHER agents (Claude Code, Codex, etc.) as sub-agents
  3. Participate in multi-agent orchestration

This module implements both sides:
  - ACPServer: Caveman acts as an ACP-compatible agent (spawnable by OpenClaw)
  - ACPClient: Caveman spawns other ACP agents via OpenClaw bridge
"""
from __future__ import annotations

import asyncio
import logging
import uuid
from datetime import datetime
from enum import Enum
from typing import Any, AsyncIterator, Callable

logger = logging.getLogger(__name__)


class ACPEventType(Enum):
    """ACP event types for streaming."""
    MESSAGE = "message"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    THINKING = "thinking"
    STATUS = "status"
    ERROR = "error"
    DONE = "done"


class ACPSession:
    """Represents an ACP agent session."""

    def __init__(self, session_id: str | None = None, agent_id: str = "caveman"):
        self.session_id = session_id or str(uuid.uuid4())[:12]
        self.agent_id = agent_id
        self.created_at = datetime.now().isoformat()
        self.messages: list[dict[str, Any]] = []
        self.status = "active"
        self._event_queue: asyncio.Queue[dict] = asyncio.Queue()

    def add_message(self, role: str, content: str, **kwargs) -> None:
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now().isoformat(),
            **kwargs,
        })

    async def emit(self, event_type: ACPEventType, data: Any) -> None:
        await self._event_queue.put({
            "type": event_type.value,
            "data": data,
            "timestamp": datetime.now().isoformat(),
        })

    async def events(self) -> AsyncIterator[dict]:
        while True:
            event = await self._event_queue.get()
            yield event
            if event["type"] == ACPEventType.DONE.value:
                break

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "agent_id": self.agent_id,
            "status": self.status,
            "created_at": self.created_at,
            "message_count": len(self.messages),
        }


class ACPServer:
    """Caveman as an ACP-compatible agent."""

    def __init__(self, agent_loop_factory: Callable[..., Any] | None = None):
        self._sessions: dict[str, ACPSession] = {}
        self._loop_factory = agent_loop_factory

    async def create_session(self, task: str, config: dict | None = None) -> ACPSession:
        session = ACPSession(agent_id="caveman")
        session.add_message("user", task)
        self._sessions[session.session_id] = session
        asyncio.create_task(self._process_session(session, task, config or {}))
        return session

    async def load_session(self, session_id: str, messages: list[dict]) -> ACPSession:
        session = self._sessions.get(session_id)
        if not session:
            session = ACPSession(session_id=session_id)
            self._sessions[session_id] = session
        for msg in messages:
            session.add_message(msg.get("role", "user"), msg.get("content", ""))
        return session

    async def get_session(self, session_id: str) -> ACPSession | None:
        return self._sessions.get(session_id)

    async def end_session(self, session_id: str) -> None:
        session = self._sessions.get(session_id)
        if session:
            session.status = "completed"
            await session.emit(ACPEventType.DONE, {"reason": "session_ended"})

    async def _process_session(self, session: ACPSession, task: str, config: dict) -> None:
        try:
            await session.emit(ACPEventType.STATUS, {"message": "Processing task..."})
            if self._loop_factory:
                loop = self._loop_factory(**config)
                result = await loop.run(task)
                session.add_message("assistant", str(result))
                await session.emit(ACPEventType.MESSAGE, {"content": str(result)})
            else:
                await session.emit(
                    ACPEventType.MESSAGE,
                    {"content": f"Task received: {task} (no agent loop configured)"},
                )
            session.status = "completed"
            await session.emit(ACPEventType.DONE, {"reason": "task_complete"})
        except Exception as e:
            session.status = "failed"
            await session.emit(ACPEventType.ERROR, {"message": str(e)})
            await session.emit(ACPEventType.DONE, {"reason": "error"})
            logger.error("ACP session %s failed: %s", session.session_id, e)


class ACPClient:
    """Spawn and communicate with other ACP agents via OpenClaw bridge.

    Fixed: sessions keyed by session_key (not agent_id) to support
    multiple instances of the same agent type.
    """

    def __init__(self, openclaw_bridge=None):
        self._bridge = openclaw_bridge
        # Key by session_key, not agent_id (audit fix: supports multiple same-type agents)
        self._sessions: dict[str, dict] = {}

    async def spawn(
        self,
        task: str,
        agent_id: str = "claude-code",
        mode: str = "run",
        timeout: int | None = None,
        sandbox: str = "inherit",
    ) -> dict:
        """Spawn an ACP agent via OpenClaw bridge."""
        if not self._bridge:
            raise RuntimeError("OpenClaw bridge not configured")

        result = await self._bridge.call_tool("sessions_spawn", {
            "task": task,
            "agentId": agent_id,
            "runtime": "acp",
            "mode": mode,
            "sandbox": sandbox,
            "runTimeoutSeconds": timeout,
        })

        # Extract session key from result (audit fix: validate response shape)
        session_key = self._extract_session_key(result, agent_id)

        session_info = {
            "session_key": session_key,
            "agent_id": agent_id,
            "task": task,
            "status": "running",
            "spawned_at": datetime.now().isoformat(),
            "raw_result": result,
        }
        # Key by session_key for uniqueness (audit fix)
        self._sessions[session_key] = session_info
        return session_info

    async def send(self, session_key: str, message: str) -> str:
        """Send a message to a spawned agent session."""
        if not self._bridge:
            raise RuntimeError("OpenClaw bridge not configured")

        result = await self._bridge.call_tool("sessions_send", {
            "sessionKey": session_key,
            "message": message,
        })
        return result.get("result", str(result))

    def list_spawned(self) -> list[dict]:
        return list(self._sessions.values())

    def get_session(self, session_key: str) -> dict | None:
        return self._sessions.get(session_key)

    @staticmethod
    def _extract_session_key(result: dict, agent_id: str) -> str:
        """Extract session key from spawn result, with fallback."""
        # Try common result shapes
        for key in ("sessionKey", "session_key", "key"):
            if key in result:
                return str(result[key])
        # Try nested result
        inner = result.get("result", "")
        if isinstance(inner, dict):
            for key in ("sessionKey", "session_key", "key"):
                if key in inner:
                    return str(inner[key])
        # Fallback: generate a unique key
        fallback = f"{agent_id}:{uuid.uuid4().hex}"
        logger.warning("Could not extract session key from spawn result, using fallback: %s", fallback)
        return fallback
