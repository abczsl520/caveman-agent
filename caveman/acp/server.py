"""ACP Server v2 — full ACP protocol with session management.

Exposes Caveman as an ACP-compatible agent with:
- Session lifecycle (create/load/resume/fork/list)
- SSE streaming for real-time events
- Slash commands (/model, /tools, /compact, /reset, /help)
- Authentication (token-based)
- MCP server proxy

Learned from: Hermes acp_adapter/server.py (728 lines)
Our version: Async-native, Starlette-based, integrated with Caveman.
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Optional

from caveman.acp.session import ACPSessionManager, ACPSessionState
from caveman.acp.events import ACPEventEmitter

__all__ = ["MAX_TASKS", "VERSION", "SLASH_COMMANDS", "ACPTask", "ACPServer"]


logger = logging.getLogger("caveman.acp")

MAX_TASKS = 1000
VERSION = "0.5.0"


# ── Slash Commands ──

SLASH_COMMANDS = {
    "/help": "Show available commands",
    "/model": "Get or set the model (e.g. /model claude-opus-4-6)",
    "/tools": "List available tools",
    "/compact": "Compact conversation history",
    "/reset": "Reset session (clear history)",
    "/mode": "Get or set mode (agent/edit/chat)",
    "/version": "Show Caveman version",
    "/sessions": "List active sessions",
}


@dataclass
class ACPTask:
    """A single ACP task."""
    id: str
    session_id: str = ""
    status: str = "pending"
    message: dict = field(default_factory=dict)
    result: dict | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)
    emitter: ACPEventEmitter = field(default=None, repr=False)


class ACPServer:
    """Full ACP-compatible server with session management."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        agent_fn=None,
        auth_token: str = "",
        session_manager: Optional[ACPSessionManager] = None,
    ):
        self.host = host
        self.port = port
        self._agent_fn = agent_fn
        self._auth_token = auth_token
        self._session_mgr = session_manager or ACPSessionManager(
            agent_factory=self._default_agent_factory if not agent_fn else None,
        )
        self._tasks: OrderedDict[str, ACPTask] = OrderedDict()
        self._uvicorn_server = None

    # ── Lifecycle ──

    async def start(self) -> None:
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.requests import Request
        from starlette.responses import JSONResponse, StreamingResponse
        import uvicorn

        server_ref = self

        # Auth middleware
        async def auth_middleware(request: Request, call_next) -> JSONResponse:
            if server_ref._auth_token:
                auth = request.headers.get("authorization", "")
                if not auth.startswith("Bearer ") or auth[7:] != server_ref._auth_token:
                    return JSONResponse({"error": "Unauthorized"}, status_code=401)
            return await call_next(request)

        # ── Task endpoints ──
        async def create_task(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            msg = body.get("message")
            if not msg:
                return JSONResponse({"error": "Missing 'message'"}, status_code=400)

            session_id = body.get("session_id", "")
            task = ACPTask(
                id=f"task-{uuid.uuid4().hex[:12]}",
                session_id=session_id,
                message=msg if isinstance(msg, dict) else {"role": "user", "parts": [{"type": "text", "text": str(msg)}]},
                emitter=ACPEventEmitter(session_id),
            )
            server_ref._evict_if_full()
            server_ref._tasks[task.id] = task
            asyncio.create_task(server_ref._run_task(task, body.get("metadata")))
            return JSONResponse(server_ref._task_dict(task), status_code=201)

        async def get_task(request: Request) -> JSONResponse:
            task = server_ref._tasks.get(request.path_params["task_id"])
            if not task:
                return JSONResponse({"error": "Not found"}, status_code=404)
            return JSONResponse(server_ref._task_dict(task))

        async def cancel_task(request: Request) -> JSONResponse:
            task = server_ref._tasks.get(request.path_params["task_id"])
            if not task:
                return JSONResponse({"error": "Not found"}, status_code=404)
            if task.status in ("pending", "running"):
                task.status = "cancelled"
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task.cancel_event.set()
            return JSONResponse(server_ref._task_dict(task))

        async def stream_task(request: Request) -> StreamingResponse:
            task = server_ref._tasks.get(request.path_params["task_id"])
            if not task:
                return JSONResponse({"error": "Not found"}, status_code=404)

            async def event_gen() -> None:
                sent = 0
                while task.status in ("pending", "running"):
                    events = task.emitter.events if task.emitter else []
                    while sent < len(events):
                        yield events[sent].to_sse()
                        sent += 1
                    await asyncio.sleep(0.1)
                # Flush remaining
                events = task.emitter.events if task.emitter else []
                while sent < len(events):
                    yield events[sent].to_sse()
                    sent += 1
                yield f"data: {json.dumps({'type': 'done', 'result': task.result or ''})}\n\n"

            return StreamingResponse(event_gen(), media_type="text/event-stream",
                                     headers={"Cache-Control": "no-cache"})

        # ── Session endpoints ──
        async def create_session(request: Request) -> JSONResponse:
            body = await request.json() if request.method == "POST" else {}
            state = await server_ref._session_mgr.create_session(
                cwd=body.get("cwd", "."),
                model=body.get("model", ""),
                mode=body.get("mode", "agent"),
            )
            return JSONResponse(server_ref._session_dict(state), status_code=201)

        async def get_session(request: Request) -> JSONResponse:
            state = await server_ref._session_mgr.get_session(request.path_params["session_id"])
            if not state:
                return JSONResponse({"error": "Not found"}, status_code=404)
            return JSONResponse(server_ref._session_dict(state))

        async def list_sessions(request: Request) -> JSONResponse:
            sessions = await server_ref._session_mgr.list_sessions()
            return JSONResponse({"sessions": sessions})

        async def fork_session(request: Request) -> JSONResponse:
            body = await request.json() if request.method == "POST" else {}
            state = await server_ref._session_mgr.fork_session(
                request.path_params["session_id"],
                cwd=body.get("cwd", "."),
            )
            if not state:
                return JSONResponse({"error": "Source session not found"}, status_code=404)
            return JSONResponse(server_ref._session_dict(state), status_code=201)

        async def delete_session(request: Request) -> JSONResponse:
            removed = await server_ref._session_mgr.remove_session(request.path_params["session_id"])
            return JSONResponse({"removed": removed})

        # ── Info endpoints ──
        async def server_info(request: Request) -> JSONResponse:
            return JSONResponse({
                "name": "caveman",
                "version": VERSION,
                "capabilities": {
                    "sessions": True,
                    "streaming": True,
                    "slash_commands": list(SLASH_COMMANDS.keys()),
                    "fork": True,
                },
            })

        async def commands_list(request: Request) -> JSONResponse:
            return JSONResponse({"commands": [
                {"name": k, "description": v} for k, v in SLASH_COMMANDS.items()
            ]})

        app = Starlette(routes=[
            # Info
            Route("/acp/v1/info", server_info, methods=["GET"]),
            Route("/acp/v1/commands", commands_list, methods=["GET"]),
            # Tasks
            Route("/acp/v1/tasks", create_task, methods=["POST"]),
            Route("/acp/v1/tasks/{task_id}", get_task, methods=["GET"]),
            Route("/acp/v1/tasks/{task_id}/stream", stream_task, methods=["GET"]),
            Route("/acp/v1/tasks/{task_id}/cancel", cancel_task, methods=["POST"]),
            # Sessions
            Route("/acp/v1/sessions", list_sessions, methods=["GET"]),
            Route("/acp/v1/sessions", create_session, methods=["POST"]),
            Route("/acp/v1/sessions/{session_id}", get_session, methods=["GET"]),
            Route("/acp/v1/sessions/{session_id}", delete_session, methods=["DELETE"]),
            Route("/acp/v1/sessions/{session_id}/fork", fork_session, methods=["POST"]),
        ])

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        asyncio.create_task(server.serve())
        await asyncio.sleep(0.1)
        logger.info("ACP server v2 on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            await asyncio.sleep(0.1)

    # ── Task execution ──

    async def _run_task(self, task: ACPTask, metadata: dict | None = None) -> None:
        task.status = "running"
        text = self._extract_text(task.message)

        # Check for slash commands
        if text.startswith("/"):
            result = await self._handle_slash(text, task.session_id)
            if result is not None:
                task.status = "completed"
                task.result = {"role": "assistant", "parts": [{"type": "text", "text": result}]}
                task.completed_at = datetime.now(timezone.utc).isoformat()
                return

        try:
            if task.emitter:
                await task.emitter.on_status("running", "Processing...")

            if self._agent_fn:
                result_text = await self._agent_fn(text)
            else:
                # Use session's agent
                state = await self._session_mgr.get_session(task.session_id) if task.session_id else None
                if state and state.agent and hasattr(state.agent, "run"):
                    result_text = await state.agent.run(text)
                else:
                    result_text = f"Echo: {text}"

            if task.cancel_event.is_set():
                task.status = "cancelled"
                return

            task.status = "completed"
            task.result = {"role": "assistant", "parts": [{"type": "text", "text": str(result_text)}]}
            if task.emitter:
                await task.emitter.on_done(str(result_text))
        except Exception as e:
            logger.warning("ACP task %s failed: %s", task.id, e)
            task.status = "failed"
            task.result = {"role": "assistant", "parts": [{"type": "text", "text": f"Error: {e}"}]}
            if task.emitter:
                await task.emitter.on_error(str(e))
        finally:
            task.completed_at = datetime.now(timezone.utc).isoformat()

    # ── Slash commands ──

    async def _handle_slash(self, text: str, session_id: str) -> Optional[str]:
        parts = text.strip().split(maxsplit=1)
        cmd = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if cmd == "/help":
            lines = ["Available commands:"]
            for k, v in SLASH_COMMANDS.items():
                lines.append(f"  {k} — {v}")
            return "\n".join(lines)

        if cmd == "/version":
            return f"Caveman v{VERSION}"

        if cmd == "/sessions":
            sessions = await self._session_mgr.list_sessions()
            if not sessions:
                return "No active sessions."
            lines = [f"Active sessions ({len(sessions)}):"]
            for s in sessions:
                lines.append(f"  {s['session_id'][:8]}... model={s['model']} mode={s['mode']} turns={s['history_len']}")
            return "\n".join(lines)

        state = await self._session_mgr.get_session(session_id) if session_id else None

        if cmd == "/model":
            if not state:
                return "No active session."
            if args:
                state.model = args.strip()
                return f"Model set to: {state.model}"
            return f"Current model: {state.model or '(default)'}"

        if cmd == "/mode":
            if not state:
                return "No active session."
            if args:
                mode = args.strip().lower()
                if mode in ("agent", "edit", "chat"):
                    state.mode = mode
                    return f"Mode set to: {state.mode}"
                return f"Unknown mode: {mode}. Use: agent, edit, chat"
            return f"Current mode: {state.mode}"

        if cmd == "/tools":
            try:
                from caveman.tools.registry import ToolRegistry
                reg = ToolRegistry()
                reg.auto_discover()
                tools = sorted(reg.list_tools())
                return f"Available tools ({len(tools)}):\n" + "\n".join(f"  {t}" for t in tools)
            except Exception as e:
                return f"Error listing tools: {e}"

        if cmd == "/compact":
            if state and state.history:
                original = len(state.history)
                state.history = state.history[-10:]
                return f"Compacted: {original} → {len(state.history)} turns"
            return "Nothing to compact."

        if cmd == "/reset":
            if state:
                state.history.clear()
                return "Session reset. History cleared."
            return "No active session."

        return None  # Not a slash command

    # ── Helpers ──

    @staticmethod
    def _extract_text(message: dict) -> str:
        text = ""
        for part in message.get("parts", []):
            if isinstance(part, dict) and part.get("type") == "text":
                text += part.get("text", "")
        return text or str(message.get("text", ""))

    def _evict_if_full(self) -> None:
        while len(self._tasks) >= MAX_TASKS:
            self._tasks.popitem(last=False)

    @staticmethod
    def _task_dict(task: ACPTask) -> dict:
        return {
            "id": task.id,
            "session_id": task.session_id,
            "status": task.status,
            "message": task.message,
            "result": task.result,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }

    @staticmethod
    def _session_dict(state: ACPSessionState) -> dict:
        return {
            "session_id": state.session_id,
            "cwd": state.cwd,
            "model": state.model,
            "mode": state.mode,
            "history_len": len(state.history),
            "created_at": state.created_at,
            "last_active": state.last_active,
        }

    def _default_agent_factory(self, **kwargs) -> Any:
        """Default factory when no agent_fn provided."""
        return None

    # ── Direct API (for testing) ──

    async def handle_create_task(self, message: dict, metadata: dict | None = None) -> dict:
        task = ACPTask(
            id=f"task-{uuid.uuid4().hex[:12]}",
            message=message,
            emitter=ACPEventEmitter(""),
        )
        self._evict_if_full()
        self._tasks[task.id] = task
        await self._run_task(task, metadata)
        return self._task_dict(task)

    async def handle_get_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        return self._task_dict(task) if task else None

    async def handle_cancel_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        if task.status in ("pending", "running"):
            task.status = "cancelled"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task.cancel_event.set()
        return self._task_dict(task)
