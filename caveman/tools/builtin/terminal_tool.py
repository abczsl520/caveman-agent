"""Terminal tool — unified command execution across environments.

Extends the basic bash tool with:
- Environment selection (local/docker/ssh) via config
- Background task management
- Automatic cleanup after inactivity
- User interrupt support
- Output streaming for long commands

This is the "full" terminal tool; bash.py is the lightweight version.
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import signal
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from caveman.paths import CAVEMAN_HOME
from caveman.tools.registry import tool

__all__ = [
    "BackgroundTask",
    "TaskRegistry",
    "execute_command",
    "terminal_tool",
    # v2 features merged in
    "_BLOCKED_PATTERNS",
    "_check_guards",
    "_truncate_output",
    "_validate_workdir",
]


logger = logging.getLogger(__name__)

_MAX_FOREGROUND_TIMEOUT = int(os.getenv("TERMINAL_MAX_FOREGROUND_TIMEOUT", "600"))
_MAX_OUTPUT = 100_000  # chars
_MAX_OUTPUT_LINES = 500

# Dangerous command patterns — block destructive system commands
_BLOCKED_PATTERNS = [
    r"\brm\s+-rf\s+/",
    r"\bmkfs\b",
    r"\bdd\s+.*of=/dev/",
    r":\(\)\{\s*:\|:&\s*\};:",  # fork bomb
    r"\bchmod\s+-R\s+777\s+/\b",
    r"\bshutdown\b",
    r"\breboot\b",
]


def _check_guards(command: str) -> Optional[str]:
    """Check command against safety guards. Returns error message or None."""
    for pattern in _BLOCKED_PATTERNS:
        if re.search(pattern, command):
            return f"Blocked: dangerous command pattern detected"
    return None


def _truncate_output(output: str) -> str:
    """Truncate output to reasonable limits, keeping head and tail."""
    lines = output.split("\n")
    total_chars = len(output)
    total_lines = len(lines)

    if total_chars <= _MAX_OUTPUT and total_lines <= _MAX_OUTPUT_LINES:
        return output

    if total_lines > _MAX_OUTPUT_LINES:
        keep = _MAX_OUTPUT_LINES // 3
        lines = lines[:keep] + [
            f"\n... ({total_lines - 2 * keep} lines omitted) ...\n"
        ] + lines[-keep:]
        output = "\n".join(lines)

    if len(output) > _MAX_OUTPUT:
        half = _MAX_OUTPUT // 2
        output = (
            output[:half]
            + f"\n... (output truncated, {total_chars} chars / {total_lines} lines total)\n"
            + output[-half:]
        )
    return output


def _validate_workdir(workdir: str) -> Optional[str]:
    """Validate working directory. Returns error or None."""
    if not workdir:
        return None
    path = Path(workdir)
    if not path.exists():
        return f"Working directory does not exist: {workdir}"
    if not path.is_dir():
        return f"Not a directory: {workdir}"
    return None


@dataclass
class BackgroundTask:
    """A long-running terminal process managed in the background."""
    task_id: str
    command: str
    pid: int
    started_at: float
    cwd: str
    output_file: Path
    status: str = "running"  # running, completed, failed, killed


class TaskRegistry:
    """Manages background terminal tasks."""

    def __init__(self) -> None:
        self._tasks: dict[str, BackgroundTask] = {}

    def register(self, task: BackgroundTask) -> None:
        self._tasks[task.task_id] = task

    def get(self, task_id: str) -> BackgroundTask | None:
        return self._tasks.get(task_id)

    def list_tasks(self) -> list[BackgroundTask]:
        self._refresh_statuses()
        return list(self._tasks.values())

    def kill(self, task_id: str) -> bool:
        task = self._tasks.get(task_id)
        if not task or task.status != "running":
            return False
        try:
            os.kill(task.pid, signal.SIGTERM)
            task.status = "killed"
            return True
        except ProcessLookupError:
            task.status = "completed"
            return False

    def get_output(self, task_id: str, tail: int = 50) -> str:
        task = self._tasks.get(task_id)
        if not task:
            return "Task not found"
        if not task.output_file.exists():
            return "(no output)"
        lines = task.output_file.read_text(encoding="utf-8", errors="replace").splitlines()
        if tail and len(lines) > tail:
            return f"... ({len(lines) - tail} lines omitted)\n" + "\n".join(lines[-tail:])
        return "\n".join(lines)

    def cleanup_old(self, max_age: float = 3600) -> int:
        """Remove completed tasks older than max_age seconds."""
        now = time.time()
        to_remove = []
        for tid, task in self._tasks.items():
            if task.status != "running" and (now - task.started_at) > max_age:
                to_remove.append(tid)
                task.output_file.unlink(missing_ok=True)
        for tid in to_remove:
            del self._tasks[tid]
        return len(to_remove)

    def _refresh_statuses(self) -> None:
        for task in self._tasks.values():
            if task.status == "running":
                try:
                    os.kill(task.pid, 0)  # Check if alive
                except ProcessLookupError:
                    task.status = "completed"


# Global registry
_registry = TaskRegistry()


async def execute_command(
    command: str,
    timeout: int = 30,
    cwd: str | None = None,
    background: bool = False,
    env: str = "local",
    extra_env: Optional[Dict[str, str]] = None,
) -> dict[str, Any]:
    """Execute a command in the specified environment.

    Args:
        command: Shell command to execute.
        timeout: Timeout in seconds (foreground only).
        cwd: Working directory.
        background: Run in background.
        env: Environment: "local", "docker", "ssh".
        extra_env: Additional environment variables.

    Returns dict with: stdout, stderr, exit_code, task_id (if background).
    """
    # Safety guards — block destructive commands
    guard_error = _check_guards(command)
    if guard_error:
        return {"stdout": "", "stderr": guard_error, "exit_code": -1}

    # Validate working directory
    if cwd:
        wd_error = _validate_workdir(cwd)
        if wd_error:
            return {"stdout": "", "stderr": wd_error, "exit_code": -1}

    timeout = min(timeout, _MAX_FOREGROUND_TIMEOUT)

    if env == "docker":
        return await _execute_docker(command, timeout, cwd)
    elif env == "ssh":
        return await _execute_ssh(command, timeout, cwd)

    if background:
        return _execute_background(command, cwd)

    return await _execute_local(command, timeout, cwd, extra_env=extra_env)


async def _execute_local(
    command: str, timeout: int, cwd: str | None,
    extra_env: Optional[Dict[str, str]] = None,
) -> dict[str, Any]:
    """Execute locally with async subprocess."""
    run_env = None
    if extra_env:
        run_env = dict(os.environ)
        run_env.update(extra_env)
    try:
        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=run_env,
        )
        stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        return {
            "stdout": _truncate_output((stdout or b"").decode(errors="replace")),
            "stderr": _truncate_output((stderr or b"").decode(errors="replace")),
            "exit_code": proc.returncode or 0,
        }
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return {"stdout": "", "stderr": f"Command timed out after {timeout}s", "exit_code": -1}
    except Exception as e:
        return {"stdout": "", "stderr": str(e), "exit_code": -1}


def _execute_background(command: str, cwd: str | None) -> dict[str, Any]:
    """Execute in background, return task_id."""
    import uuid
    task_id = f"bg-{uuid.uuid4().hex[:12]}"
    output_file = CAVEMAN_HOME / "tasks" / f"{task_id}.log"
    output_file.parent.mkdir(parents=True, exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        proc = subprocess.Popen(
            command, shell=True, stdout=f, stderr=subprocess.STDOUT,
            cwd=cwd, start_new_session=True,
        )

    task = BackgroundTask(
        task_id=task_id,
        command=command,
        pid=proc.pid,
        started_at=time.time(),
        cwd=cwd or os.getcwd(),
        output_file=output_file,
    )
    _registry.register(task)

    return {
        "stdout": f"Background task started: {task_id} (PID {proc.pid})",
        "stderr": "",
        "exit_code": 0,
        "task_id": task_id,
    }


async def _execute_docker(command: str, timeout: int, cwd: str | None) -> dict[str, Any]:
    """Execute in Docker container."""
    try:
        from caveman.environments import DockerEnv
        docker = DockerEnv()
        result = await docker.run(command, timeout=timeout)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}
    except (ImportError, Exception) as e:
        return {"stdout": "", "stderr": f"Docker environment error: {e}", "exit_code": -1}


async def _execute_ssh(command: str, timeout: int, cwd: str | None) -> dict[str, Any]:
    """Execute via SSH."""
    try:
        from caveman.environments import SSHEnv
        ssh = SSHEnv()
        result = await ssh.run(command, timeout=timeout)
        return {"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code}
    except (ImportError, Exception) as e:
        return {"stdout": "", "stderr": f"SSH environment error: {e}", "exit_code": -1}


@tool(
    name="terminal",
    description="Execute commands with safety guards, environment selection, "
    "background task support, and output truncation.",
    params={
        "command": {"type": "string", "description": "Shell command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30)"},
        "workdir": {"type": "string", "description": "Working directory (optional)"},
        "background": {"type": "boolean", "description": "Run in background"},
        "env": {"type": "string", "description": "Environment: local/docker/ssh"},
        "task_action": {"type": "string", "description": "Task action: list/output/kill"},
        "task_id": {"type": "string", "description": "Task ID for task actions"},
    },
    required=[],
)
async def terminal_tool(
    command: str = "",
    timeout: int = 30,
    workdir: str = "",
    background: bool = False,
    env: str = "local",
    task_action: str = "",
    task_id: str = "",
) -> str:
    """Terminal tool for agent use."""
    # Task management actions
    if task_action == "list":
        tasks = _registry.list_tasks()
        if not tasks:
            return "No background tasks"
        lines = [f"  {t.task_id}: {t.status} (PID {t.pid}) — {t.command[:60]}" for t in tasks]
        return "Background tasks:\n" + "\n".join(lines)

    if task_action == "output" and task_id:
        return _registry.get_output(task_id)

    if task_action == "kill" and task_id:
        if _registry.kill(task_id):
            return f"Killed task {task_id}"
        return f"Could not kill task {task_id}"

    if not command:
        return "No command specified"

    result = await execute_command(
        command, timeout=timeout, cwd=workdir or None,
        background=background, env=env,
    )

    parts = []
    if result.get("task_id"):
        parts.append(result["stdout"])
    else:
        if result["stdout"]:
            parts.append(result["stdout"])
        if result["stderr"]:
            parts.append(f"STDERR: {result['stderr']}")
        parts.append(f"Exit code: {result['exit_code']}")

    return "\n".join(parts)
