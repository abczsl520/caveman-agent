"""Remote execution environments — SSH, Docker, local.

Provides a unified interface for executing commands in different environments.
The agent doesn't need to know where code runs — it just calls execute().
"""
from __future__ import annotations

import asyncio
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any
from caveman.aio import aio_exists, aio_mkdir, aio_read_text, aio_write_text

__all__ = ["Environment", "LocalEnv", "SSHEnv", "DockerEnv", "create_env"]

logger = logging.getLogger(__name__)


@dataclass
class ExecResult:
    """Result of a command execution."""
    stdout: str = ""
    stderr: str = ""
    returncode: int = 0
    timed_out: bool = False

    @property
    def success(self) -> bool:
        return self.returncode == 0 and not self.timed_out


class Environment(ABC):
    """Abstract execution environment."""

    @abstractmethod
    async def execute(self, command: str, timeout: int = 60, cwd: str | None = None) -> ExecResult: ...

    @abstractmethod
    async def read_file(self, path: str) -> str: ...

    @abstractmethod
    async def write_file(self, path: str, content: str) -> None: ...

    @abstractmethod
    async def file_exists(self, path: str) -> bool: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    async def setup(self) -> None:
        """Optional setup hook."""

    async def teardown(self) -> None:
        """Optional teardown hook."""


class LocalEnv(Environment):
    """Local execution environment."""

    @property
    def name(self) -> str:
        return "local"

    async def execute(self, command: str, timeout: int = 60, cwd: str | None = None) -> ExecResult:
        try:
            proc = await asyncio.create_subprocess_shell(
                command, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
                cwd=cwd,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                stdout=stdout.decode(errors="replace")[:100_000],
                stderr=stderr.decode(errors="replace")[:50_000],
                returncode=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(stderr="Command timed out", returncode=-1, timed_out=True)

    async def read_file(self, path: str) -> str:
        return await aio_read_text(path, encoding="utf-8")

    async def write_file(self, path: str, content: str) -> None:
        from pathlib import Path
        await aio_mkdir(Path(path).parent, parents=True, exist_ok=True)
        await aio_write_text(Path(path), content, encoding="utf-8")

    async def file_exists(self, path: str) -> bool:
        return await aio_exists(path)


@dataclass
class SSHConfig:
    """SSH connection configuration."""
    host: str
    user: str = "root"
    port: int = 22
    key_file: str | None = None
    password: str | None = None
    connect_timeout: int = 10


class SSHEnv(Environment):
    """Remote execution via SSH."""

    def __init__(self, config: SSHConfig):
        self.config = config
        self._ssh_base: list[str] = []

    @property
    def name(self) -> str:
        return f"ssh:{self.config.user}@{self.config.host}"

    async def setup(self) -> None:
        """Build SSH command base."""
        parts = [
            "ssh", "-o", "StrictHostKeyChecking=no",
            "-o", f"ConnectTimeout={self.config.connect_timeout}",
            "-p", str(self.config.port),
        ]
        if self.config.key_file:
            parts.extend(["-i", self.config.key_file])
        parts.append(f"{self.config.user}@{self.config.host}")
        self._ssh_base = parts
        # Test connection
        result = await self.execute("echo ok", timeout=15)
        if not result.success:
            raise ConnectionError(f"SSH connection failed: {result.stderr}")

    async def execute(self, command: str, timeout: int = 60, cwd: str | None = None) -> ExecResult:
        if not self._ssh_base:
            await self.setup()
        cmd = command
        if cwd:
            cmd = f"cd {cwd} && {command}"
        full_cmd = self._ssh_base + [cmd]
        try:
            proc = await asyncio.create_subprocess_exec(
                *full_cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                stdout=stdout.decode(errors="replace")[:100_000],
                stderr=stderr.decode(errors="replace")[:50_000],
                returncode=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            proc.kill()
            return ExecResult(stderr="SSH command timed out", returncode=-1, timed_out=True)

    async def read_file(self, path: str) -> str:
        result = await self.execute(f"cat {path}")
        if not result.success:
            raise FileNotFoundError(f"Remote file not found: {path} ({result.stderr})")
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        import shlex
        escaped = shlex.quote(content)
        await self.execute(f"mkdir -p $(dirname {path}) && echo {escaped} > {path}")

    async def file_exists(self, path: str) -> bool:
        result = await self.execute(f"test -f {path} && echo yes || echo no")
        return result.stdout.strip() == "yes"


class DockerEnv(Environment):
    """Docker container execution environment."""

    def __init__(self, image: str = "python:3.12-slim", container_name: str | None = None):
        self.image = image
        self.container_name = container_name or f"caveman-env-{id(self)}"
        self._running = False

    @property
    def name(self) -> str:
        return f"docker:{self.image}"

    async def setup(self) -> None:
        proc = await asyncio.create_subprocess_exec(
            "docker", "run", "-d", "--name", self.container_name,
            "--rm", self.image, "sleep", "infinity",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            raise RuntimeError(f"Docker setup failed: {stderr.decode()}")
        self._running = True

    async def teardown(self) -> None:
        if self._running:
            proc = await asyncio.create_subprocess_exec(
                "docker", "stop", self.container_name,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            await proc.communicate()
            self._running = False

    async def execute(self, command: str, timeout: int = 60, cwd: str | None = None) -> ExecResult:
        cmd = command
        if cwd:
            cmd = f"cd {cwd} && {command}"
        try:
            proc = await asyncio.create_subprocess_exec(
                "docker", "exec", self.container_name, "sh", "-c", cmd,
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return ExecResult(
                stdout=stdout.decode(errors="replace")[:100_000],
                stderr=stderr.decode(errors="replace")[:50_000],
                returncode=proc.returncode or 0,
            )
        except asyncio.TimeoutError:
            return ExecResult(stderr="Docker command timed out", returncode=-1, timed_out=True)

    async def read_file(self, path: str) -> str:
        result = await self.execute(f"cat {path}")
        if not result.success:
            raise FileNotFoundError(f"Container file not found: {path}")
        return result.stdout

    async def write_file(self, path: str, content: str) -> None:
        import shlex
        escaped = shlex.quote(content)
        await self.execute(f"mkdir -p $(dirname {path}) && echo {escaped} > {path}")

    async def file_exists(self, path: str) -> bool:
        result = await self.execute(f"test -f {path} && echo yes || echo no")
        return result.stdout.strip() == "yes"


def create_env(config: dict[str, Any] | None = None) -> Environment:
    """Factory: create environment from config."""
    if not config:
        return LocalEnv()
    env_type = config.get("type", "local")
    if env_type == "local":
        return LocalEnv()
    elif env_type == "ssh":
        return SSHEnv(SSHConfig(
            host=config["host"],
            user=config.get("user", "root"),
            port=config.get("port", 22),
            key_file=config.get("key_file"),
        ))
    elif env_type == "docker":
        return DockerEnv(
            image=config.get("image", "python:3.12-slim"),
            container_name=config.get("container_name"),
        )
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
