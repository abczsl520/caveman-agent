"""Health check endpoint — lightweight HTTP server for monitoring.

Provides /health and /status endpoints for:
- Container orchestration (k8s liveness/readiness probes)
- External monitoring (uptime checks)
- Gateway diagnostics

Runs on a separate port (default 4201) to avoid interfering with the main gateway.
"""
from __future__ import annotations
import asyncio
import json
import logging
import time
from caveman.timeouts import DRAIN_LONG

logger = logging.getLogger(__name__)

DEFAULT_HEALTH_PORT = 4201


class HealthServer:
    """Minimal HTTP health check server."""

    def __init__(self, port: int = DEFAULT_HEALTH_PORT, status_fn=None):
        self.port = port
        self._status_fn = status_fn or (lambda: {})
        self._server: asyncio.AbstractServer | None = None
        self._start_time = time.time()

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=DRAIN_LONG)
            line = data.decode("utf-8", errors="replace").strip()
            path = line.split(" ")[1] if " " in line else "/"

            if path == "/health":
                body = json.dumps({"status": "ok", "uptime_s": round(time.time() - self._start_time)})
                status = "200 OK"
            elif path == "/status":
                body = json.dumps(self._status_fn(), default=str)
                status = "200 OK"
            else:
                body = json.dumps({"error": "not found"})
                status = "404 Not Found"

            response = (
                f"HTTP/1.1 {status}\r\n"
                f"Content-Type: application/json\r\n"
                f"Content-Length: {len(body)}\r\n"
                f"\r\n{body}"
            )
            writer.write(response.encode())
            await writer.drain()
        except Exception as exc:
            logger.debug("_handle: suppressed %s", exc)
        finally:
            writer.close()

    async def start(self) -> None:
        try:
            self._server = await asyncio.start_server(
                self._handle, "0.0.0.0", self.port)
            logger.info("Health check server on port %d", self.port)
        except OSError as e:
            logger.warning("Health server failed to start: %s", e)

    async def stop(self) -> None:
        if self._server:
            self._server.close()
            await self._server.wait_closed()
            self._server = None
