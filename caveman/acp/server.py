"""ACP Server — expose Caveman as an ACP-compatible agent.

ACP is a JSON-over-HTTP protocol for agent-to-agent communication:
  POST /acp/v1/tasks       — Create a new task
  GET  /acp/v1/tasks/{id}  — Get task status/result
  POST /acp/v1/tasks/{id}/cancel — Cancel a task
"""
from __future__ import annotations

import asyncio
import json
import logging
import uuid
from collections import OrderedDict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any

logger = logging.getLogger(__name__)

MAX_TASKS = 1000


@dataclass
class ACPTask:
    """A single ACP task."""

    id: str
    status: str = "pending"  # pending, running, completed, failed, cancelled
    message: dict = field(default_factory=dict)
    result: dict | None = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    completed_at: str | None = None
    _cancel_event: asyncio.Event = field(default_factory=asyncio.Event, repr=False)


class ACPServer:
    """ACP-compatible HTTP server that runs agent tasks."""

    def __init__(
        self,
        host: str = "127.0.0.1",
        port: int = 8766,
        agent_fn=None,
    ):
        """
        Args:
            host: Bind address.
            port: Bind port.
            agent_fn: Async callable(message: str) -> str that runs the agent.
                      If None, tasks echo back the input.
        """
        self.host = host
        self.port = port
        self._agent_fn = agent_fn
        self._tasks: OrderedDict[str, ACPTask] = OrderedDict()
        self._server: asyncio.Server | None = None

    # ── Lifecycle ──

    async def start(self) -> None:
        from starlette.applications import Starlette
        from starlette.routing import Route
        from starlette.requests import Request
        from starlette.responses import JSONResponse
        import uvicorn

        async def create_task(request: Request) -> JSONResponse:
            try:
                body = await request.json()
            except Exception:
                return JSONResponse({"error": "Invalid JSON"}, status_code=400)
            msg = body.get("message")
            if not msg or not isinstance(msg, dict):
                return JSONResponse({"error": "Missing 'message' field"}, status_code=400)
            task = ACPTask(id=f"task-{uuid.uuid4().hex[:12]}", message=msg)
            self._evict_if_full()
            self._tasks[task.id] = task
            asyncio.create_task(self._run_task(task, body.get("metadata")))
            return JSONResponse(self._task_to_dict(task), status_code=201)

        async def get_task(request: Request) -> JSONResponse:
            task_id = request.path_params["task_id"]
            task = self._tasks.get(task_id)
            if not task:
                return JSONResponse({"error": "Task not found"}, status_code=404)
            return JSONResponse(self._task_to_dict(task))

        async def cancel_task(request: Request) -> JSONResponse:
            task_id = request.path_params["task_id"]
            task = self._tasks.get(task_id)
            if not task:
                return JSONResponse({"error": "Task not found"}, status_code=404)
            if task.status not in ("completed", "failed", "cancelled"):
                task.status = "cancelled"
                task.completed_at = datetime.now(timezone.utc).isoformat()
                task._cancel_event.set()
            return JSONResponse(self._task_to_dict(task))

        async def stream_task(request: Request) -> Response:
            """SSE endpoint — stream task events in real-time."""
            task_id = request.path_params["task_id"]
            task = self._tasks.get(task_id)
            if not task:
                return JSONResponse({"error": "Task not found"}, status_code=404)

            async def event_generator():
                import json as _json
                if hasattr(task, "_stream_events"):
                    for evt in task._stream_events:
                        yield f"data: {_json.dumps(evt.to_dict())}\n\n"
                yield f"data: {_json.dumps({'type': 'done', 'data': task.result or ''})}\n\n"

            from starlette.responses import StreamingResponse
            return StreamingResponse(
                event_generator(),
                media_type="text/event-stream",
                headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
            )

        app = Starlette(routes=[
            Route("/acp/v1/tasks", create_task, methods=["POST"]),
            Route("/acp/v1/tasks/{task_id}", get_task, methods=["GET"]),
            Route("/acp/v1/tasks/{task_id}/stream", stream_task, methods=["GET"]),
            Route("/acp/v1/tasks/{task_id}/cancel", cancel_task, methods=["POST"]),
        ])

        config = uvicorn.Config(app, host=self.host, port=self.port, log_level="warning")
        server = uvicorn.Server(config)
        self._uvicorn_server = server
        # Run in background task so start() returns
        asyncio.create_task(server.serve())
        # Give it a moment to bind
        await asyncio.sleep(0.1)
        logger.info("ACP server listening on %s:%s", self.host, self.port)

    async def stop(self) -> None:
        if hasattr(self, "_uvicorn_server") and self._uvicorn_server:
            self._uvicorn_server.should_exit = True
            await asyncio.sleep(0.1)

    # ── Task execution ──

    async def _run_task(self, task: ACPTask, metadata: dict | None = None) -> None:
        task.status = "running"
        text = ""
        for part in task.message.get("parts", []):
            if part.get("type") == "text":
                text += part.get("text", "")

        try:
            if self._agent_fn:
                result_text = await self._agent_fn(text)
            else:
                result_text = f"Echo: {text}"

            if task.status == "cancelled":
                return

            task.status = "completed"
            task.result = {
                "role": "assistant",
                "parts": [{"type": "text", "text": result_text}],
            }
        except Exception as e:
            logger.warning("ACP task %s failed: %s", task.id, e)
            task.status = "failed"
            task.result = {
                "role": "assistant",
                "parts": [{"type": "text", "text": f"Error: {e}"}],
            }
        finally:
            task.completed_at = datetime.now(timezone.utc).isoformat()

    def _evict_if_full(self) -> None:
        while len(self._tasks) >= MAX_TASKS:
            self._tasks.popitem(last=False)

    @staticmethod
    def _task_to_dict(task: ACPTask) -> dict:
        return {
            "id": task.id,
            "status": task.status,
            "message": task.message,
            "result": task.result,
            "created_at": task.created_at,
            "completed_at": task.completed_at,
        }

    # ── Direct API (for testing without HTTP) ──

    async def handle_create_task(self, message: dict, metadata: dict | None = None) -> dict:
        """Create a task programmatically (no HTTP)."""
        task = ACPTask(id=f"task-{uuid.uuid4().hex[:12]}", message=message)
        self._evict_if_full()
        self._tasks[task.id] = task
        await self._run_task(task, metadata)
        return self._task_to_dict(task)

    async def handle_get_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        return self._task_to_dict(task) if task else None

    async def handle_cancel_task(self, task_id: str) -> dict | None:
        task = self._tasks.get(task_id)
        if not task:
            return None
        if task.status not in ("completed", "failed", "cancelled"):
            task.status = "cancelled"
            task.completed_at = datetime.now(timezone.utc).isoformat()
            task._cancel_event.set()
        return self._task_to_dict(task)
