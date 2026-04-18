"""Sandbox — isolated code execution environment.

Security model (honest about limitations):
  - Process isolation via subprocess (NOT container/chroot/seccomp)
  - Path containment for preloaded files (validated)
  - Process group kill on timeout (prevents orphan persistence)
  - Restricted PATH and HOME
  - Network: advisory only (env var, not enforced — documented limitation)

What this sandbox does NOT provide:
  - Filesystem isolation (code CAN read host files)
  - True network blocking
  - Memory/CPU cgroup limits
  - Syscall filtering

For true isolation, use Docker or a VM. This sandbox is a "best effort"
defense-in-depth layer, not a security boundary.
"""
from __future__ import annotations

import asyncio
import logging
import os
import shutil
import signal
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SandboxResult:
    """Result from sandboxed execution."""
    stdout: str
    stderr: str
    exit_code: int
    duration: float
    timed_out: bool
    sandbox_dir: str


@dataclass
class SandboxConfig:
    """Sandbox configuration."""
    timeout: int = 30
    max_output_bytes: int = 100_000  # 100KB (reduced from 1MB to limit DoS)
    allow_network: bool = False
    max_memory_mb: int = 256
    cleanup: bool = True
    allowed_commands: list[str] = field(default_factory=lambda: [
        "python3", "python", "node",
    ])  # Removed bash/sh — too dangerous for untrusted code


class Sandbox:
    """Isolated execution environment for untrusted code.

    WARNING: This is NOT a true security sandbox. See module docstring.
    """

    def __init__(self, config: SandboxConfig | None = None) -> None:
        self.config = config or SandboxConfig()

    async def execute(
        self,
        code: str,
        language: str = "python",
        files: dict[str, str] | None = None,
    ) -> SandboxResult:
        """Execute code in a restricted subprocess."""
        sandbox_dir = tempfile.mkdtemp(prefix="caveman_sandbox_")

        try:
            # Write code file
            ext = {"python": ".py", "javascript": ".js", "bash": ".sh"}.get(language, ".txt")
            code_file = Path(sandbox_dir) / f"main{ext}"
            code_file.write_text(code, encoding="utf-8")

            # Write additional files — with path containment
            if files:
                for name, content in files.items():
                    safe_path = _resolve_safe_path(sandbox_dir, name)
                    if safe_path is None:
                        return SandboxResult(
                            stdout="",
                            stderr=f"⛔ Path traversal blocked: {name}",
                            exit_code=1, duration=0, timed_out=False,
                            sandbox_dir=sandbox_dir,
                        )
                    safe_path.parent.mkdir(parents=True, exist_ok=True)
                    safe_path.write_text(content, encoding="utf-8")

            # Build command
            cmd = self._build_command(language, str(code_file))
            if not cmd:
                return SandboxResult(
                    stdout="", stderr=f"Unsupported language: {language}",
                    exit_code=1, duration=0, timed_out=False,
                    sandbox_dir=sandbox_dir,
                )

            env = self._build_env(sandbox_dir)
            start = time.monotonic()
            timed_out = False

            try:
                # Start in new process group for clean kill
                proc = await asyncio.create_subprocess_exec(
                    *cmd,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE,
                    cwd=sandbox_dir,
                    env=env,
                    start_new_session=True,  # P1 #5 fix: new session for group kill
                )

                try:
                    stdout_bytes, stderr_bytes = await asyncio.wait_for(
                        proc.communicate(), timeout=self.config.timeout,
                    )
                except asyncio.TimeoutError:
                    timed_out = True
                    # Kill entire process group, not just the leader
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGTERM)
                    except (ProcessLookupError, PermissionError, OSError):
                        pass
                    await asyncio.sleep(1)
                    try:
                        pgid = os.getpgid(proc.pid)
                        os.killpg(pgid, signal.SIGKILL)
                    except (ProcessLookupError, PermissionError, OSError):
                        try:
                            proc.kill()
                        except (ProcessLookupError, OSError):
                            pass
                    try:
                        await asyncio.wait_for(proc.communicate(), timeout=3)
                    except (asyncio.TimeoutError, ProcessLookupError, OSError):
                        pass
                    stdout_bytes = b"[timed out]"
                    stderr_bytes = b""

                duration = time.monotonic() - start

                max_b = self.config.max_output_bytes
                stdout = stdout_bytes[:max_b].decode("utf-8", errors="replace")
                stderr = stderr_bytes[:max_b].decode("utf-8", errors="replace")
                exit_code = proc.returncode if proc.returncode is not None else -1

            except FileNotFoundError:
                duration = time.monotonic() - start
                stdout = ""
                stderr = f"Interpreter not found: {cmd[0]}"
                exit_code = 127

            return SandboxResult(
                stdout=stdout, stderr=stderr, exit_code=exit_code,
                duration=duration, timed_out=timed_out, sandbox_dir=sandbox_dir,
            )

        finally:
            if self.config.cleanup:
                shutil.rmtree(sandbox_dir, ignore_errors=True)

    def _build_command(self, language: str, code_file: str) -> list[str] | None:
        interpreters = {
            "python": ["python3", "-u", code_file],
            "javascript": ["node", code_file],
            "bash": ["bash", code_file],
            "sh": ["sh", code_file],
        }
        cmd = interpreters.get(language)
        if not cmd:
            return None
        binary = cmd[0]
        if binary not in self.config.allowed_commands:
            return None
        if not shutil.which(binary):
            return None
        return cmd

    def _build_env(self, sandbox_dir: str) -> dict[str, str]:
        env = {
            "HOME": sandbox_dir,
            "TMPDIR": sandbox_dir,
            "PATH": "/usr/local/bin:/usr/bin:/bin",
            "LANG": "en_US.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "CAVEMAN_SANDBOX": "1",
        }
        if not self.config.allow_network:
            env["CAVEMAN_NO_NETWORK"] = "1"
        return env

    async def validate_skill(self, skill_code: str) -> tuple[bool, str]:
        """Validate skill syntax safely."""
        # P2 #13 fix: use repr() to safely embed code, not triple-quote
        escaped = repr(skill_code)
        test_code = (
            "import ast\n"
            "try:\n"
            f"    ast.parse({escaped})\n"
            "    print('VALID')\n"
            "except SyntaxError as e:\n"
            "    print(f'INVALID: {e}')\n"
        )
        result = await self.execute(test_code, language="python")
        valid = "VALID" in result.stdout and "INVALID" not in result.stdout
        return valid, result.stdout.strip()


def _resolve_safe_path(base_dir: str, name: str) -> Path | None:
    """Resolve a file path safely within base_dir.

    Returns None if the path would escape the base directory.
    """
    base = Path(base_dir).resolve()
    # Reject absolute paths
    if os.path.isabs(name):
        logger.warning("Path traversal blocked (absolute): %s", name)
        return None
    target = (base / name).resolve()
    # Ensure target is under base
    try:
        target.relative_to(base)
    except ValueError:
        logger.warning("Path traversal blocked (escape): %s → %s", name, target)
        return None
    return target
