"""CLI agent runner — delegate coding tasks to external AI agents.

Supports Claude Code, Codex, and other CLI-based coding agents
via subprocess invocation with timeout and output capture.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import time
from dataclasses import dataclass
from typing import Any

from caveman.errors import CavemanError

logger = logging.getLogger(__name__)


class CLIAgentError(CavemanError):
    """CLI agent execution error."""
    pass


@dataclass(frozen=True, slots=True)
class CLIAgentResult:
    """Result from a CLI agent invocation."""
    output: str
    exit_code: int
    duration: float
    timed_out: bool
    agent: str


# Agent definitions — cmd, pty requirement, default timeout
_AGENT_DEFS: dict[str, dict[str, Any]] = {
    "claude": {
        "cmd": ["claude", "--print", "--permission-mode", "bypassPermissions", "-p"],
        "pty": False,
        "timeout": 900,
        "description": "Claude Code — non-interactive, one-shot output",
    },
    "codex": {
        "cmd": ["codex", "exec"],
        "pty": True,
        "timeout": 900,
        "description": "Codex CLI — interactive, requires PTY",
    },
    "gemini": {
        "cmd": ["gemini"],
        "pty": True,
        "timeout": 900,
        "description": "Gemini CLI — interactive, requires PTY",
    },
    "pi": {
        "cmd": ["opencode"],
        "pty": True,
        "timeout": 900,
        "description": "Pi (OpenCode) — interactive, requires PTY",
    },
    "aider": {
        "cmd": ["aider", "--yes", "--no-git", "--message"],
        "pty": False,
        "timeout": 600,
        "description": "Aider — non-interactive with --message flag",
    },
}


class CLIAgentRunner:
    """Run external coding agents as subprocesses."""

    def __init__(self, agents: dict[str, dict[str, Any]] | None = None) -> None:
        self._agents = agents or dict(_AGENT_DEFS)

    def available(self) -> list[str]:
        """Check which agents are installed (on PATH)."""
        result = []
        for name, defn in self._agents.items():
            binary = defn["cmd"][0]
            if shutil.which(binary):
                result.append(name)
        return result

    async def run(
        self,
        agent: str,
        task: str,
        cwd: str | None = None,
        timeout: int | None = None,
    ) -> CLIAgentResult:
        """Run a CLI agent and capture output.

        Automatically uses PTY for agents that require it (codex, gemini, pi).
        """
        if agent not in self._agents:
            raise CLIAgentError(
                f"Unknown agent: {agent!r}",
                context={"agent": agent, "known": list(self._agents.keys())},
            )

        defn = self._agents[agent]
        use_pty = defn.get("pty", False)

        if use_pty:
            return await self._run_pty(agent, defn, task, cwd, timeout)
        return await self._run_pipe(agent, defn, task, cwd, timeout)

    async def _run_pipe(
        self, agent: str, defn: dict, task: str,
        cwd: str | None, timeout: int | None,
    ) -> CLIAgentResult:
        """Run agent via pipe (non-interactive)."""
        cmd = defn["cmd"] + [task]
        agent_timeout = timeout or defn.get("timeout", 900)
        work_dir = cwd or os.getcwd()

        start = time.monotonic()
        timed_out = False

        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.STDOUT,
                cwd=work_dir,
            )
            try:
                stdout, _ = await asyncio.wait_for(
                    proc.communicate(), timeout=agent_timeout,
                )
            except asyncio.TimeoutError:
                timed_out = True
                proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.communicate(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()
                    await proc.communicate()
                stdout = b"[timed out]"

            duration = time.monotonic() - start
            output = stdout.decode("utf-8", errors="replace") if stdout else ""
            exit_code = proc.returncode if proc.returncode is not None else -1

        except FileNotFoundError:
            duration = time.monotonic() - start
            output = f"Agent binary not found: {defn['cmd'][0]}"
            exit_code = 127
        except Exception as e:
            duration = time.monotonic() - start
            output = f"Error running agent: {e}"
            exit_code = -1

        return CLIAgentResult(
            output=output, exit_code=exit_code,
            duration=duration, timed_out=timed_out, agent=agent,
        )

    async def _run_pty(
        self, agent: str, defn: dict, task: str,
        cwd: str | None, timeout: int | None,
    ) -> CLIAgentResult:
        """Run agent via PTY (interactive agents like codex/gemini/pi)."""
        import pty
        import select

        cmd = defn["cmd"] + [task]
        agent_timeout = timeout or defn.get("timeout", 900)
        work_dir = cwd or os.getcwd()

        start = time.monotonic()
        timed_out = False
        output_chunks: list[bytes] = []
        master_fd = -1
        slave_fd = -1

        try:
            # Create PTY pair
            master_fd, slave_fd = pty.openpty()

            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=work_dir,
            )
            # Close slave immediately after spawn (child has its own copy)
            os.close(slave_fd)
            slave_fd = -1

            # Read output from master_fd with timeout
            deadline = time.monotonic() + agent_timeout

            while True:
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    timed_out = True
                    break

                # Check if process has exited
                if proc.returncode is not None:
                    # Drain remaining output
                    while True:
                        r, _, _ = select.select([master_fd], [], [], 0.1)
                        if not r:
                            break
                        try:
                            chunk = os.read(master_fd, 4096)
                            if not chunk:
                                break
                            output_chunks.append(chunk)
                        except OSError:
                            break
                    break

                # Non-blocking read
                r, _, _ = select.select([master_fd], [], [], min(1.0, remaining))
                if r:
                    try:
                        chunk = os.read(master_fd, 4096)
                        if not chunk:
                            break
                        output_chunks.append(chunk)
                    except OSError:
                        break
                else:
                    # Check if process exited during wait (safe check, no internal access)
                    if proc.returncode is not None:
                        continue  # Loop back to drain

            if timed_out:
                proc.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(proc.wait(), timeout=10)
                except asyncio.TimeoutError:
                    proc.kill()

            duration = time.monotonic() - start
            output = b"".join(output_chunks).decode("utf-8", errors="replace")
            exit_code = proc.returncode if proc.returncode is not None else -1

        except FileNotFoundError:
            duration = time.monotonic() - start
            output = f"Agent binary not found: {defn['cmd'][0]}"
            exit_code = 127
        except Exception as e:
            duration = time.monotonic() - start
            output = f"Error running agent: {e}"
            exit_code = -1
        finally:
            # Audit fix: always close fds to prevent leaks on any exception path
            if master_fd >= 0:
                try:
                    os.close(master_fd)
                except OSError:
                    pass
            if slave_fd >= 0:
                try:
                    os.close(slave_fd)
                except OSError:
                    pass

        return CLIAgentResult(
            output=output, exit_code=exit_code,
            duration=duration, timed_out=timed_out, agent=agent,
        )
