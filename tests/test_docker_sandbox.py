"""Tests for Docker sandbox."""
import pytest
from caveman.sandbox import (
    SandboxConfig, is_docker_available, run_in_docker, run_sandboxed,
    DEFAULT_IMAGE, DEFAULT_TIMEOUT, DEFAULT_MEMORY_LIMIT,
)


class TestSandboxConfig:
    def test_defaults(self):
        cfg = SandboxConfig()
        assert cfg.image == DEFAULT_IMAGE
        assert cfg.timeout == DEFAULT_TIMEOUT
        assert cfg.memory_limit == DEFAULT_MEMORY_LIMIT
        assert cfg.network_mode == "none"
        assert cfg.workspace_mount is None

    def test_custom(self):
        cfg = SandboxConfig(image="node:20", timeout=30, network_mode="bridge")
        assert cfg.image == "node:20"
        assert cfg.timeout == 30
        assert cfg.network_mode == "bridge"


class TestDockerAvailability:
    @pytest.mark.asyncio
    async def test_check_docker(self):
        """Just verify the check doesn't crash."""
        result = await is_docker_available()
        assert isinstance(result, bool)


class TestRunSandboxed:
    @pytest.mark.asyncio
    async def test_python_code(self):
        """Test Python execution (uses subprocess fallback if no Docker)."""
        result = await run_sandboxed("print('hello')")
        assert result["ok"]
        assert "hello" in result["stdout"]

    @pytest.mark.asyncio
    async def test_python_error(self):
        result = await run_sandboxed("raise ValueError('boom')")
        assert not result["ok"]
        assert "boom" in result["stderr"]

    @pytest.mark.asyncio
    async def test_timeout(self):
        result = await run_sandboxed(
            "import time; time.sleep(10)",
            config=SandboxConfig(timeout=2),
        )
        assert not result["ok"]

    @pytest.mark.asyncio
    async def test_non_python_without_docker(self):
        """Non-Python requires Docker."""
        if await is_docker_available():
            pytest.skip("Docker is available, can't test fallback")
        result = await run_sandboxed("console.log('hi')", language="node")
        assert not result["ok"]
        assert "Docker required" in result["stderr"]
