"""Docker sandbox — isolated code execution in containers.

Provides secure execution environment with:
  - Network isolation (configurable)
  - Filesystem isolation (workspace mounts)
  - Resource limits (CPU, memory, timeout)
  - Auto-cleanup of containers
  - Fallback to subprocess sandbox if Docker unavailable
"""
from __future__ import annotations

import asyncio
import logging
import shutil
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any
from caveman.aio import aio_write_text

__all__ = ["SandboxConfig", "is_docker_available", "run_in_docker", "run_sandboxed"]

logger = logging.getLogger(__name__)

# --- Constants ---

DEFAULT_IMAGE = "python:3.12-slim"
DEFAULT_TIMEOUT = 60
DEFAULT_MEMORY_LIMIT = "512m"
DEFAULT_CPU_LIMIT = "1.0"
MAX_OUTPUT = 100 * 1024  # 100KB


@dataclass
class SandboxConfig:
    """Configuration for Docker sandbox."""
    image: str = DEFAULT_IMAGE
    timeout: int = DEFAULT_TIMEOUT
    memory_limit: str = DEFAULT_MEMORY_LIMIT
    cpu_limit: str = DEFAULT_CPU_LIMIT
    network_mode: str = "none"  # none, bridge, host
    workspace_mount: str | None = None  # host path to mount as /workspace
    extra_mounts: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)


async def is_docker_available() -> bool:
    """Check if Docker daemon is running."""
    try:
        proc = await asyncio.create_subprocess_exec(
            "docker", "info",
            stdout=asyncio.subprocess.DEVNULL,
            stderr=asyncio.subprocess.DEVNULL,
        )
        code = await asyncio.wait_for(proc.wait(), timeout=5)
        return code == 0
    except (FileNotFoundError, asyncio.TimeoutError, OSError):
        return False


async def run_in_docker(
    code: str,
    language: str = "python",
    config: SandboxConfig | None = None,
) -> dict[str, Any]:
    """Execute code in a Docker container.

    Returns: {"ok": bool, "stdout": str, "stderr": str, "exit_code": int, "duration": float}
    """
    cfg = config or SandboxConfig()

    # Write code to temp file
    tmp = tempfile.mkdtemp(prefix="caveman-sandbox-")
    try:
        ext = {"python": ".py", "bash": ".sh", "node": ".js", "javascript": ".js"}.get(language, ".txt")
        code_file = Path(tmp) / f"run{ext}"
        await aio_write_text(code_file, code, encoding="utf-8")

        # Build docker run command
        cmd = ["docker", "run", "--rm"]

        # Resource limits
        cmd.extend(["--memory", cfg.memory_limit])
        cmd.extend(["--cpus", cfg.cpu_limit])

        # Network
        cmd.extend(["--network", cfg.network_mode])

        # Security
        cmd.extend(["--security-opt", "no-new-privileges"])
        cmd.extend(["--read-only"])
        cmd.extend(["--tmpfs", "/tmp:rw,noexec,nosuid,size=64m"])

        # Mount code
        cmd.extend(["-v", f"{tmp}:/code:ro"])

        # Workspace mount
        if cfg.workspace_mount:
            cmd.extend(["-v", f"{cfg.workspace_mount}:/workspace:rw"])

        # Extra mounts
        for mount in cfg.extra_mounts:
            cmd.extend(["-v", mount])

        # Environment
        cmd.extend(["-e", "CAVEMAN_SANDBOX=1"])
        for k, v in cfg.env.items():
            cmd.extend(["-e", f"{k}={v}"])

        # Image and command
        cmd.append(cfg.image)

        if language == "python":
            cmd.extend(["python", f"/code/run{ext}"])
        elif language == "bash":
            cmd.extend(["bash", f"/code/run{ext}"])
        elif language in ("node", "javascript"):
            cmd.extend(["node", f"/code/run{ext}"])
        else:
            cmd.extend(["cat", f"/code/run{ext}"])

        import time as _time
        start = _time.monotonic()

        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            stdout, stderr = await asyncio.wait_for(
                proc.communicate(), timeout=cfg.timeout
            )
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()
            return {
                "ok": False,
                "stdout": "",
                "stderr": f"Timeout after {cfg.timeout}s",
                "exit_code": -1,
                "duration": _time.monotonic() - start,
                "engine": "docker",
            }

        duration = _time.monotonic() - start
        out = stdout.decode("utf-8", errors="replace")[:MAX_OUTPUT]
        err = stderr.decode("utf-8", errors="replace")[:MAX_OUTPUT]

        return {
            "ok": proc.returncode == 0,
            "stdout": out,
            "stderr": err,
            "exit_code": proc.returncode,
            "duration": round(duration, 2),
            "engine": "docker",
        }

    finally:
        shutil.rmtree(tmp, ignore_errors=True)


async def run_sandboxed(
    code: str,
    language: str = "python",
    config: SandboxConfig | None = None,
) -> dict[str, Any]:
    """Run code in Docker if available, otherwise fall back to subprocess sandbox."""
    if await is_docker_available():
        return await run_in_docker(code, language, config)

    # Fallback to subprocess sandbox
    logger.info("Docker not available, falling back to subprocess sandbox")
    if language != "python":
        return {
            "ok": False,
            "stdout": "",
            "stderr": f"Subprocess sandbox only supports Python. Docker required for {language}.",
            "exit_code": -1,
            "duration": 0,
            "engine": "subprocess",
        }

    # Use existing subprocess sandbox
    from caveman.tools.builtin.sandbox_tool import sandbox_exec
    result = await sandbox_exec(code, timeout=config.timeout if config else DEFAULT_TIMEOUT)
    result["engine"] = "subprocess"
    return result
