"""Tests for execution environments."""
import pytest
from caveman.environments import (
    LocalEnv, SSHEnv, SSHConfig, DockerEnv, create_env, ExecResult,
)


class TestExecResult:
    def test_success(self):
        r = ExecResult(stdout="ok", returncode=0)
        assert r.success

    def test_failure(self):
        r = ExecResult(stderr="err", returncode=1)
        assert not r.success

    def test_timeout(self):
        r = ExecResult(timed_out=True)
        assert not r.success


class TestLocalEnv:
    @pytest.mark.asyncio
    async def test_execute(self):
        env = LocalEnv()
        result = await env.execute("echo hello")
        assert result.success
        assert "hello" in result.stdout

    @pytest.mark.asyncio
    async def test_execute_failure(self):
        env = LocalEnv()
        result = await env.execute("false")
        assert not result.success

    @pytest.mark.asyncio
    async def test_execute_timeout(self):
        env = LocalEnv()
        result = await env.execute("sleep 10", timeout=1)
        assert result.timed_out

    @pytest.mark.asyncio
    async def test_read_write_file(self, tmp_path):
        env = LocalEnv()
        path = str(tmp_path / "test.txt")
        await env.write_file(path, "hello world")
        content = await env.read_file(path)
        assert "hello world" in content

    @pytest.mark.asyncio
    async def test_file_exists(self, tmp_path):
        env = LocalEnv()
        assert not await env.file_exists(str(tmp_path / "nope.txt"))
        (tmp_path / "yes.txt").write_text("x")
        assert await env.file_exists(str(tmp_path / "yes.txt"))

    def test_name(self):
        assert LocalEnv().name == "local"


class TestSSHEnv:
    def test_name(self):
        env = SSHEnv(SSHConfig(host="example.com", user="admin"))
        assert env.name == "ssh:admin@example.com"

    def test_config_defaults(self):
        cfg = SSHConfig(host="1.2.3.4")
        assert cfg.user == "root"
        assert cfg.port == 22


class TestDockerEnv:
    def test_name(self):
        env = DockerEnv(image="node:20")
        assert env.name == "docker:node:20"


class TestCreateEnv:
    def test_default_local(self):
        env = create_env()
        assert isinstance(env, LocalEnv)

    def test_explicit_local(self):
        env = create_env({"type": "local"})
        assert isinstance(env, LocalEnv)

    def test_ssh(self):
        env = create_env({"type": "ssh", "host": "1.2.3.4", "user": "deploy"})
        assert isinstance(env, SSHEnv)
        assert env.config.host == "1.2.3.4"

    def test_docker(self):
        env = create_env({"type": "docker", "image": "ubuntu:22.04"})
        assert isinstance(env, DockerEnv)
        assert env.image == "ubuntu:22.04"

    def test_unknown_type(self):
        with pytest.raises(ValueError, match="Unknown"):
            create_env({"type": "quantum"})
