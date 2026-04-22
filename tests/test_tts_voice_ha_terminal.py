"""Tests for TTS, Voice Mode, Home Assistant, and Terminal tools."""
import pytest
from unittest.mock import patch, AsyncMock, MagicMock
from pathlib import Path

from caveman.tools.builtin.tts_tool import TTSProvider, _DEFAULT_VOICES, list_edge_voices
from caveman.tools.builtin.voice_mode import VoiceMode, VoiceRecorder, AudioPlayer, _audio_available
from caveman.tools.builtin.homeassistant_tool import _BLOCKED_DOMAINS, _ENTITY_ID_RE
from caveman.tools.builtin.terminal_tool import TaskRegistry, BackgroundTask, execute_command


# ── TTS Tests ──

class TestTTSProvider:
    def test_all_providers_have_default_voice(self):
        for provider in TTSProvider:
            assert provider in _DEFAULT_VOICES

    def test_provider_values(self):
        assert TTSProvider.EDGE.value == "edge"
        assert TTSProvider.OPENAI.value == "openai"
        assert TTSProvider.ELEVENLABS.value == "elevenlabs"
        assert TTSProvider.FISH.value == "fish"


# ── Voice Mode Tests ──

class TestVoiceMode:
    def test_check_requirements(self):
        vm = VoiceMode()
        reqs = vm.check_requirements()
        assert "audio_libs" in reqs
        assert "ffmpeg" in reqs

    def test_recorder_initial_state(self):
        recorder = VoiceRecorder()
        assert recorder.is_recording is False


# ── Home Assistant Tests ──

class TestHomeAssistant:
    def test_blocked_domains(self):
        assert "shell_command" in _BLOCKED_DOMAINS
        assert "hassio" in _BLOCKED_DOMAINS
        assert "light" not in _BLOCKED_DOMAINS

    def test_entity_id_regex(self):
        assert _ENTITY_ID_RE.match("light.living_room")
        assert _ENTITY_ID_RE.match("sensor.temperature_1")
        assert not _ENTITY_ID_RE.match("invalid")
        assert not _ENTITY_ID_RE.match("LIGHT.room")
        assert not _ENTITY_ID_RE.match("light.")

    @pytest.mark.asyncio
    async def test_call_service_blocked_domain(self):
        from caveman.tools.builtin.homeassistant_tool import ha_call_service
        result = await ha_call_service("shell_command", "run", "sensor.test")
        assert "blocked" in result


# ── Terminal Tool Tests ──

class TestTaskRegistry:
    def test_register_and_list(self):
        reg = TaskRegistry()
        task = BackgroundTask(
            task_id="test-1", command="echo hi", pid=99999,
            started_at=1000, cwd="/tmp", output_file=Path("/tmp/test.log"),
        )
        reg.register(task)
        tasks = reg.list_tasks()
        assert len(tasks) == 1
        assert tasks[0].task_id == "test-1"

    def test_get_task(self):
        reg = TaskRegistry()
        task = BackgroundTask(
            task_id="test-1", command="echo", pid=99999,
            started_at=1000, cwd="/tmp", output_file=Path("/tmp/test.log"),
        )
        reg.register(task)
        assert reg.get("test-1") is not None
        assert reg.get("nonexistent") is None

    def test_get_output(self, tmp_path):
        log_file = tmp_path / "test.log"
        log_file.write_text("line1\nline2\nline3\n")
        reg = TaskRegistry()
        task = BackgroundTask(
            task_id="test-1", command="echo", pid=99999,
            started_at=1000, cwd="/tmp", output_file=log_file,
        )
        reg.register(task)
        output = reg.get_output("test-1")
        assert "line1" in output

    def test_get_output_tail(self, tmp_path):
        log_file = tmp_path / "test.log"
        lines = [f"line{i}" for i in range(100)]
        log_file.write_text("\n".join(lines))
        reg = TaskRegistry()
        task = BackgroundTask(
            task_id="test-1", command="echo", pid=99999,
            started_at=1000, cwd="/tmp", output_file=log_file,
        )
        reg.register(task)
        output = reg.get_output("test-1", tail=10)
        assert "omitted" in output

    def test_cleanup_old(self):
        import time
        reg = TaskRegistry()
        task = BackgroundTask(
            task_id="old", command="echo", pid=99999,
            started_at=time.time() - 7200, cwd="/tmp",
            output_file=Path("/tmp/nonexistent.log"),
            status="completed",
        )
        reg.register(task)
        removed = reg.cleanup_old(max_age=3600)
        assert removed == 1
        assert len(reg.list_tasks()) == 0


class TestExecuteCommand:
    @pytest.mark.asyncio
    async def test_local_echo(self):
        result = await execute_command("echo hello", timeout=5)
        assert "hello" in result["stdout"]
        assert result["exit_code"] == 0

    @pytest.mark.asyncio
    async def test_local_timeout(self):
        result = await execute_command("sleep 10", timeout=1)
        assert result["exit_code"] == -1

    @pytest.mark.asyncio
    async def test_local_error(self):
        result = await execute_command("false", timeout=5)
        assert result["exit_code"] != 0

    @pytest.mark.asyncio
    async def test_background(self):
        result = await execute_command("echo bg", background=True)
        assert "task_id" in result
        assert result["exit_code"] == 0
