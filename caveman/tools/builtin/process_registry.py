"""Process Registry — background process management with PTY support.

Extracted from Hermes process_registry.py (1172 lines).
Key patterns: spawn, poll, read log, write stdin, kill, session tracking.
"""
from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

__all__ = ["ProcessSession", "ProcessRegistry"]


logger = logging.getLogger("caveman.tools.process")


@dataclass
class ProcessSession:
    """A managed background process."""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    command: str = ""
    cwd: str = ""
    pid: Optional[int] = None
    status: str = "pending"  # pending | running | completed | failed | killed
    exit_code: Optional[int] = None
    output_buffer: str = ""
    error_buffer: str = ""
    started_at: float = 0
    completed_at: float = 0
    _process: Optional[asyncio.subprocess.Process] = field(default=None, repr=False)
    _task: Optional[asyncio.Task] = field(default=None, repr=False)

    @property
    def is_alive(self) -> bool:
        return self.status == "running" and self._process is not None

    @property
    def duration_ms(self) -> float:
        end = self.completed_at or time.monotonic()
        return (end - self.started_at) * 1000 if self.started_at else 0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "command": self.command[:200],
            "status": self.status,
            "pid": self.pid,
            "exit_code": self.exit_code,
            "duration_ms": round(self.duration_ms, 1),
            "output_lines": self.output_buffer.count("\n"),
        }


class ProcessRegistry:
    """Manages background processes with lifecycle tracking."""

    def __init__(self, max_sessions: int = 50, max_output: int = 100000):
        self._sessions: Dict[str, ProcessSession] = {}
        self._max_sessions = max_sessions
        self._max_output = max_output

    # ── Spawn ──

    async def spawn(
        self, command: str, cwd: str = "", env: Optional[Dict[str, str]] = None,
        timeout: float = 0,
    ) -> ProcessSession:
        """Spawn a background process."""
        self._evict_old()

        session = ProcessSession(command=command, cwd=cwd or os.getcwd())
        self._sessions[session.id] = session

        run_env = dict(os.environ)
        if env:
            run_env.update(env)

        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=session.cwd,
                env=run_env,
            )
            session._process = proc
            session.pid = proc.pid
            session.status = "running"
            session.started_at = time.monotonic()

            # Start output reader
            session._task = asyncio.create_task(
                self._reader(session, timeout)
            )

            logger.info("Spawned process %s (pid=%s): %s", session.id, proc.pid, command[:100])
            return session

        except Exception as e:
            session.status = "failed"
            session.error_buffer = str(e)
            return session

    async def _reader(self, session: ProcessSession, timeout: float) -> None:
        """Read process output until completion."""
        proc = session._process
        if not proc:
            return

        try:
            if timeout > 0:
                stdout, stderr = await asyncio.wait_for(
                    proc.communicate(), timeout=timeout
                )
            else:
                stdout, stderr = await proc.communicate()

            session.output_buffer = stdout.decode("utf-8", errors="replace")[:self._max_output]
            session.error_buffer = stderr.decode("utf-8", errors="replace")[:self._max_output]
            session.exit_code = proc.returncode
            session.status = "completed" if proc.returncode == 0 else "failed"

        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            session.status = "killed"
            session.error_buffer += "\nProcess killed: timeout"

        except Exception as e:
            session.status = "failed"
            session.error_buffer += f"\nReader error: {e}"

        finally:
            session.completed_at = time.monotonic()

    # ── Query ──

    def get(self, session_id: str) -> Optional[ProcessSession]:
        return self._sessions.get(session_id)

    def poll(self, session_id: str) -> Dict[str, Any]:
        """Poll process status and recent output."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        return {
            **session.to_dict(),
            "output_tail": session.output_buffer[-5000:] if session.output_buffer else "",
            "error_tail": session.error_buffer[-2000:] if session.error_buffer else "",
        }

    def read_log(self, session_id: str, offset: int = 0, limit: int = 200) -> Dict[str, Any]:
        """Read process output log with pagination."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}

        lines = session.output_buffer.split("\n")
        total = len(lines)
        selected = lines[offset:offset + limit]

        return {
            "lines": selected,
            "total_lines": total,
            "offset": offset,
            "has_more": offset + limit < total,
        }

    # ── Control ──

    async def kill(self, session_id: str) -> Dict[str, Any]:
        """Kill a running process."""
        session = self._sessions.get(session_id)
        if not session:
            return {"error": f"Session {session_id} not found"}
        if not session.is_alive:
            return {"error": "Process not running", "status": session.status}

        try:
            if session._process:
                session._process.kill()
                await session._process.wait()
            session.status = "killed"
            session.completed_at = time.monotonic()
            return {"ok": True, "status": "killed"}
        except Exception as e:
            return {"error": str(e)}

    async def write_stdin(self, session_id: str, data: str) -> Dict[str, Any]:
        """Write data to process stdin."""
        session = self._sessions.get(session_id)
        if not session or not session.is_alive:
            return {"error": "Process not running"}
        if not session._process or not session._process.stdin:
            return {"error": "No stdin available"}

        try:
            session._process.stdin.write(data.encode())
            await session._process.stdin.drain()
            return {"ok": True, "bytes_written": len(data)}
        except Exception as e:
            return {"error": str(e)}

    # ── Listing ──

    def list_sessions(self, status: Optional[str] = None) -> List[Dict[str, Any]]:
        sessions = list(self._sessions.values())
        if status:
            sessions = [s for s in sessions if s.status == status]
        return [s.to_dict() for s in sessions]

    # ── Cleanup ──

    def _evict_old(self) -> None:
        """Remove oldest completed sessions if at capacity."""
        if len(self._sessions) < self._max_sessions:
            return
        completed = [
            (sid, s) for sid, s in self._sessions.items()
            if s.status in ("completed", "failed", "killed")
        ]
        completed.sort(key=lambda x: x[1].completed_at)
        for sid, _ in completed[:len(completed) // 2]:
            del self._sessions[sid]

    async def cleanup_all(self) -> int:
        """Kill all running processes and clear registry."""
        killed = 0
        for session in list(self._sessions.values()):
            if session.is_alive:
                await self.kill(session.id)
                killed += 1
        self._sessions.clear()
        return killed
