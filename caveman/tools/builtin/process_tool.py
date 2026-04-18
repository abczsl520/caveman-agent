"""Process tools — start, list, read output, kill background processes."""
from __future__ import annotations

import asyncio
import logging

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

# Module-level store for background processes
_PROCESSES: dict[int, dict] = {}


@tool(
    name="process_start",
    description="Start a background process",
    params={
        "command": {"type": "string", "description": "Shell command to run"},
        "label": {"type": "string", "description": "Human-readable label"},
    },
    required=["command", "label"],
)
async def process_start(command: str, label: str) -> dict:
    """Start a background subprocess and track it."""
    proc = await asyncio.create_subprocess_shell(
        command,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
        start_new_session=True,  # Isolate process group for clean kill
    )
    _PROCESSES[proc.pid] = {"proc": proc, "label": label, "command": command}
    return {"pid": proc.pid, "label": label}


@tool(
    name="process_list",
    description="List tracked background processes",
    params={},
    required=[],
)
async def process_list() -> list[dict]:
    """List all tracked background processes."""
    return [
        {
            "pid": pid,
            "label": info["label"],
            "running": info["proc"].returncode is None,
        }
        for pid, info in _PROCESSES.items()
    ]


@tool(
    name="process_output",
    description="Get output from a background process",
    params={
        "pid": {"type": "integer", "description": "Process ID"},
        "lines": {"type": "integer", "description": "Max lines to return", "default": 50},
    },
    required=["pid"],
)
async def process_output(pid: int, lines: int = 50) -> dict:
    """Read stdout/stderr from a tracked process."""
    info = _PROCESSES.get(pid)
    if not info:
        return {"error": f"No tracked process with pid {pid}"}
    proc = info["proc"]
    running = proc.returncode is None
    stdout = b""
    stderr = b""
    if not running:
        stdout = await proc.stdout.read() if proc.stdout else b""
        stderr = await proc.stderr.read() if proc.stderr else b""
    else:
        # Non-blocking read of available data
        try:
            stdout = await asyncio.wait_for(proc.stdout.read(65536), timeout=0.5) if proc.stdout else b""
        except asyncio.TimeoutError:
            stdout = b""
        try:
            stderr = await asyncio.wait_for(proc.stderr.read(65536), timeout=0.5) if proc.stderr else b""
        except asyncio.TimeoutError:
            stderr = b""
    out_lines = stdout.decode(errors="replace").splitlines()[-lines:]
    err_lines = stderr.decode(errors="replace").splitlines()[-lines:]
    return {
        "stdout": "\n".join(out_lines),
        "stderr": "\n".join(err_lines),
        "running": running,
    }


@tool(
    name="process_kill",
    description="Kill a tracked background process",
    params={
        "pid": {"type": "integer", "description": "Process ID"},
    },
    required=["pid"],
)
async def process_kill(pid: int) -> dict:
    """Kill a tracked background process."""
    info = _PROCESSES.get(pid)
    if not info:
        return {"error": f"No tracked process with pid {pid}"}
    proc = info["proc"]
    if proc.returncode is None:
        proc.kill()
        await proc.wait()
    del _PROCESSES[pid]
    return {"ok": True}
