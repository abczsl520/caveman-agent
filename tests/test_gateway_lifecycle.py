"""Tests for gateway process lifecycle: self-kill protection, PID management,
graceful restart, and service installation.

Covers:
- Layer 1: bash self-kill protection
- Layer 2: PID file + process status
- Layer 3: SIGUSR1 graceful restart
- Layer 4: launchd/systemd service install
- Layer 5: Drain mechanism
- Layer 6: Restart sentinel
"""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════════════════
# Layer 1: Bash self-kill protection
# ═══════════════════════════════════════════════════════════════════════════

class TestBashSelfKillProtection:
    """Test that the bash tool blocks commands that would kill the gateway."""

    def test_kill_own_pid_blocked(self):
        from caveman.tools.builtin.bash import _is_self_kill
        pid = os.getpid()
        result = _is_self_kill(f"kill {pid}")
        assert result is not None
        assert "own process tree" in result

    def test_kill_own_pid_with_signal_blocked(self):
        from caveman.tools.builtin.bash import _is_self_kill
        pid = os.getpid()
        result = _is_self_kill(f"kill -9 {pid}")
        assert result is not None

    def test_kill_other_pid_allowed(self):
        from caveman.tools.builtin.bash import _is_self_kill
        # PID 99999999 is almost certainly not in our process tree
        result = _is_self_kill("kill 99999999")
        assert result is None

    def test_pkill_caveman_blocked(self):
        from caveman.tools.builtin.bash import _is_self_kill
        result = _is_self_kill("pkill caveman")
        assert result is not None
        assert "caveman" in result

    def test_killall_gateway_blocked(self):
        from caveman.tools.builtin.bash import _is_self_kill
        result = _is_self_kill("killall gateway")
        assert result is not None

    def test_pkill_unrelated_allowed(self):
        from caveman.tools.builtin.bash import _is_self_kill
        result = _is_self_kill("pkill firefox")
        assert result is None

    def test_kill_in_chained_command_blocked(self):
        from caveman.tools.builtin.bash import _is_self_kill
        pid = os.getpid()
        result = _is_self_kill(f"echo hello; kill {pid}")
        assert result is not None

    def test_kill_command_detection_regex(self):
        from caveman.tools.builtin.bash import _KILL_COMMANDS
        assert _KILL_COMMANDS.search("kill 123")
        assert _KILL_COMMANDS.search("pkill python")
        assert _KILL_COMMANDS.search("killall node")
        assert _KILL_COMMANDS.search("sudo kill -9 123")
        assert not _KILL_COMMANDS.search("echo kill")  # "kill" not at command position

    def test_indirect_kill_blocked(self):
        """Bypass vectors via subshell/xargs/python should be detected."""
        from caveman.tools.builtin.bash import _is_self_kill
        # xargs kill targeting caveman
        result = _is_self_kill("pgrep caveman | xargs kill")
        assert result is not None

        # bash -c with kill targeting gateway
        result = _is_self_kill('bash -c "kill $(pgrep gateway)"')
        assert result is not None

        # python -c with os.kill targeting self
        result = _is_self_kill('python3 -c "import os; os.kill(os.getppid(), 9)"')
        assert result is not None

    def test_indirect_kill_unrelated_allowed(self):
        """Indirect kill not targeting caveman should be allowed."""
        from caveman.tools.builtin.bash import _is_self_kill
        # xargs kill targeting something else
        result = _is_self_kill("pgrep firefox | xargs kill")
        assert result is None

    def test_process_tree_pids_includes_self(self):
        from caveman.tools.builtin.bash import _get_process_tree_pids
        pids = _get_process_tree_pids()
        assert os.getpid() in pids

    @pytest.mark.asyncio
    async def test_bash_exec_blocks_self_kill(self):
        from caveman.tools.builtin.bash import bash_exec
        pid = os.getpid()
        result = await bash_exec(f"kill {pid}")
        assert not result["success"]
        assert "Blocked" in result["stderr"]
        assert "/restart" in result["stderr"]

    @pytest.mark.asyncio
    async def test_bash_exec_allows_safe_kill(self):
        """kill targeting a non-existent PID should not be blocked by self-kill check."""
        from caveman.tools.builtin.bash import bash_exec
        result = await bash_exec("kill -0 99999999", timeout=5)
        # Should not be blocked by self-kill protection
        # (will fail because PID doesn't exist, but that's fine)
        assert "Blocked" not in result.get("stderr", "")


# ═══════════════════════════════════════════════════════════════════════════
# Layer 2: PID file + process status
# ═══════════════════════════════════════════════════════════════════════════

class TestGatewayStatus:
    """Test PID file management and process status tracking."""

    @pytest.fixture
    def tmp_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr("caveman.gateway.status.CAVEMAN_HOME", tmp_path)
        return tmp_path

    def test_write_and_read_pid_file(self, tmp_home):
        from caveman.gateway.status import write_pid_file, _read_json, _pid_path
        write_pid_file()
        data = _read_json(_pid_path())
        assert data is not None
        assert data["pid"] == os.getpid()
        assert data["kind"] == "caveman-gateway"
        assert "argv" in data
        assert "started_at" in data

    def test_remove_pid_file(self, tmp_home):
        from caveman.gateway.status import write_pid_file, remove_pid_file, _pid_path
        write_pid_file()
        assert _pid_path().exists()
        remove_pid_file()
        assert not _pid_path().exists()

    def test_remove_nonexistent_pid_file(self, tmp_home):
        from caveman.gateway.status import remove_pid_file
        # Should not raise
        remove_pid_file()
        assert True  # Idempotent removal succeeded

    def test_get_running_pid_returns_none_when_no_file(self, tmp_home):
        from caveman.gateway.status import get_running_pid
        assert get_running_pid() is None

    def test_get_running_pid_cleans_stale(self, tmp_home):
        from caveman.gateway.status import _write_json, _pid_path, get_running_pid
        # Write a PID file for a dead process
        _write_json(_pid_path(), {"pid": 99999999, "kind": "caveman-gateway"})
        assert get_running_pid() is None
        assert not _pid_path().exists()  # Cleaned up

    def test_is_gateway_running(self, tmp_home):
        from caveman.gateway.status import is_gateway_running
        assert not is_gateway_running()

    def test_looks_like_gateway_excludes_flywheel(self):
        """Flywheel process should NOT be detected as gateway."""
        from caveman.gateway.status import _looks_like_gateway
        with patch("caveman.gateway.status._get_process_cmdline",
                   return_value="caveman flywheel --target memory"):
            assert not _looks_like_gateway(12345)

    def test_looks_like_gateway_matches_serve(self):
        from caveman.gateway.status import _looks_like_gateway
        with patch("caveman.gateway.status._get_process_cmdline",
                   return_value="caveman serve --config /etc/caveman.yaml"):
            assert _looks_like_gateway(12345)

    def test_looks_like_gateway_matches_run_gateway(self):
        from caveman.gateway.status import _looks_like_gateway
        with patch("caveman.gateway.status._get_process_cmdline",
                   return_value="python -c run_gateway_forever"):
            assert _looks_like_gateway(12345)

    def test_write_runtime_state(self, tmp_home):
        from caveman.gateway.status import write_runtime_state, read_runtime_state
        write_runtime_state(state="running")
        data = read_runtime_state()
        assert data is not None
        assert data["state"] == "running"
        assert data["pid"] == os.getpid()

    def test_atomic_write_no_temp_files_left(self, tmp_home):
        """Atomic write should not leave .tmp files on success."""
        from caveman.gateway.status import write_runtime_state
        write_runtime_state(state="running")
        tmp_files = list(tmp_home.glob("*.tmp"))
        assert tmp_files == [], f"Leftover temp files: {tmp_files}"

    def test_runtime_state_update(self, tmp_home):
        from caveman.gateway.status import write_runtime_state, read_runtime_state
        write_runtime_state(state="starting")
        write_runtime_state(state="running", active_sessions=3)
        data = read_runtime_state()
        assert data["state"] == "running"
        assert data["active_sessions"] == 3

    def test_runtime_state_platform(self, tmp_home):
        from caveman.gateway.status import write_runtime_state, read_runtime_state
        write_runtime_state(state="running", platform="discord", platform_state="connected")
        data = read_runtime_state()
        assert data["platforms"]["discord"]["state"] == "connected"

    def test_terminate_pid_sends_signal(self):
        from caveman.gateway.status import terminate_pid
        with patch("caveman.gateway.status.os.kill") as mock_kill:
            terminate_pid(12345, force=False)
            mock_kill.assert_called_once_with(12345, signal.SIGTERM)

    def test_terminate_pid_force(self):
        from caveman.gateway.status import terminate_pid
        with patch("caveman.gateway.status.os.kill") as mock_kill:
            terminate_pid(12345, force=True)
            mock_kill.assert_called_once_with(12345, signal.SIGKILL)


# ═══════════════════════════════════════════════════════════════════════════
# Layer 3: Graceful restart
# ═══════════════════════════════════════════════════════════════════════════

class TestGracefulRestart:
    """Test SIGUSR1-based graceful restart mechanism."""

    def test_restart_exit_code(self):
        from caveman.gateway.restart import RESTART_EXIT_CODE
        assert RESTART_EXIT_CODE == 75

    def test_parse_drain_timeout_default(self):
        from caveman.gateway.restart import parse_drain_timeout, DEFAULT_DRAIN_TIMEOUT
        assert parse_drain_timeout(None) == DEFAULT_DRAIN_TIMEOUT
        assert parse_drain_timeout("") == DEFAULT_DRAIN_TIMEOUT
        assert parse_drain_timeout("invalid") == DEFAULT_DRAIN_TIMEOUT

    def test_parse_drain_timeout_custom(self):
        from caveman.gateway.restart import parse_drain_timeout
        assert parse_drain_timeout(60) == 60.0
        assert parse_drain_timeout("120") == 120.0
        assert parse_drain_timeout(-5) == 0.0  # Clamped to 0

    @pytest.mark.asyncio
    async def test_trigger_restart_cooldown(self):
        from caveman.gateway import restart
        # Force a recent restart timestamp
        restart._last_restart_at = time.monotonic()
        result = await restart.trigger_restart()
        assert not result["ok"]
        assert result["method"] == "cooldown"
        # Reset
        restart._last_restart_at = 0

    def test_request_restart_via_signal(self):
        from caveman.gateway.restart import request_restart_via_signal
        with patch("caveman.gateway.restart.os.kill") as mock_kill:
            result = request_restart_via_signal(pid=12345)
            assert result is True
            mock_kill.assert_called_once_with(12345, signal.SIGUSR1)

    def test_request_restart_via_signal_failure(self):
        from caveman.gateway.restart import request_restart_via_signal
        with patch("caveman.gateway.restart.os.kill", side_effect=ProcessLookupError):
            result = request_restart_via_signal(pid=99999999)
            assert result is False


# ═══════════════════════════════════════════════════════════════════════════
# Layer 6: Restart sentinel
# ═══════════════════════════════════════════════════════════════════════════

class TestRestartSentinel:
    """Test restart sentinel write/consume cycle."""

    @pytest.fixture
    def tmp_home(self, tmp_path, monkeypatch):
        monkeypatch.setattr("caveman.paths.CAVEMAN_HOME", tmp_path)
        return tmp_path

    def test_write_and_consume_sentinel(self, tmp_home):
        from caveman.gateway.restart import write_restart_sentinel, consume_restart_sentinel
        write_restart_sentinel(kind="restart", reason="test restart")
        sentinel = consume_restart_sentinel()
        assert sentinel is not None
        assert sentinel["kind"] == "restart"
        assert sentinel["reason"] == "test restart"
        assert sentinel["version"] == 1

        # Second consume should return None (file deleted)
        assert consume_restart_sentinel() is None

    def test_consume_nonexistent_sentinel(self, tmp_home):
        from caveman.gateway.restart import consume_restart_sentinel
        assert consume_restart_sentinel() is None

    def test_sentinel_with_session_key(self, tmp_home):
        from caveman.gateway.restart import write_restart_sentinel, consume_restart_sentinel
        write_restart_sentinel(kind="config-apply", reason="config change",
                               session_key="discord:123:456")
        sentinel = consume_restart_sentinel()
        assert sentinel["session_key"] == "discord:123:456"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 4: Service installation
# ═══════════════════════════════════════════════════════════════════════════

class TestServiceInstallation:
    """Test launchd/systemd service generation."""

    def test_generate_launchd_plist(self):
        from caveman.gateway.service import generate_launchd_plist, LAUNCHD_LABEL
        plist = generate_launchd_plist()
        assert LAUNCHD_LABEL in plist
        assert "caveman" in plist
        assert "serve" in plist
        assert "KeepAlive" in plist
        assert "SuccessfulExit" in plist
        assert "RunAtLoad" in plist

    def test_launchd_plist_minimal_path(self):
        """Plist PATH should be minimal, not a snapshot of current PATH."""
        from caveman.gateway.service import generate_launchd_plist
        plist = generate_launchd_plist()
        # Extract PATH value
        import re
        path_match = re.search(r'<key>PATH</key>\s*<string>(.*?)</string>', plist)
        assert path_match, "PATH not found in plist"
        path_value = path_match.group(1)
        assert len(path_value) < 200, f"PATH too long ({len(path_value)} chars): {path_value[:100]}..."
        assert "/usr/bin" in path_value

    def test_launchd_plist_user_log_path(self):
        """Plist log paths should be under CAVEMAN_HOME, not /tmp/."""
        from caveman.gateway.service import generate_launchd_plist
        plist = generate_launchd_plist()
        assert "/tmp/" not in plist, "Plist should not use /tmp/ for logs"
        assert "logs/gateway-stdout.log" in plist
        assert "logs/gateway-stderr.log" in plist

    def test_generate_systemd_unit(self):
        from caveman.gateway.service import generate_systemd_unit, SYSTEMD_SERVICE
        unit = generate_systemd_unit()
        assert "caveman serve" in unit
        assert "Restart=on-failure" in unit
        assert "RestartForceExitStatus=75" in unit
        assert "RestartSec=5" in unit

    def test_find_caveman_bin(self):
        from caveman.gateway.service import _find_caveman_bin
        result = _find_caveman_bin()
        assert result  # Should find something
        assert "caveman" in result

    @pytest.mark.skipif(sys.platform != "darwin", reason="macOS only")
    def test_launchd_plist_path(self):
        from caveman.gateway.service import _launchd_plist_path
        path = _launchd_plist_path()
        assert "LaunchAgents" in str(path)
        assert path.suffix == ".plist"

    @pytest.mark.skipif(sys.platform != "linux", reason="Linux only")
    def test_systemd_unit_path(self):
        from caveman.gateway.service import _systemd_unit_path
        path = _systemd_unit_path()
        assert "systemd" in str(path)
        assert path.suffix == ".service"


# ═══════════════════════════════════════════════════════════════════════════
# Layer 5: Drain mechanism
# ═══════════════════════════════════════════════════════════════════════════

class TestDrainMechanism:
    """Test session drain before restart."""

    @pytest.mark.asyncio
    async def test_drain_no_active_sessions(self):
        from caveman.gateway.gateway_lifecycle import drain_active_sessions
        count, timed_out = await drain_active_sessions({}, {}, 1.0)
        assert count == 0
        assert not timed_out


# ═══════════════════════════════════════════════════════════════════════════
# Integration: /restart command
# ═══════════════════════════════════════════════════════════════════════════

class TestRestartCommand:
    """Test the /restart command handler."""

    @pytest.mark.asyncio
    async def test_restart_no_gateway(self):
        from caveman.commands.handlers.system import handle_restart
        from caveman.commands.types import CommandContext

        responses = []
        ctx = MagicMock(spec=CommandContext)
        ctx.respond = lambda msg: responses.append(msg)
        ctx.t = lambda en, zh: zh  # Use Chinese

        with patch("caveman.gateway.status.get_running_pid", return_value=None):
            await handle_restart(ctx)

        assert any("没有找到" in r for r in responses)

    @pytest.mark.asyncio
    async def test_restart_success(self):
        from caveman.commands.handlers.system import handle_restart
        from caveman.commands.types import CommandContext

        responses = []
        ctx = MagicMock(spec=CommandContext)
        ctx.respond = lambda msg: responses.append(msg)
        ctx.t = lambda en, zh: zh

        async def mock_trigger():
            return {"ok": True, "method": "launchd"}

        with patch("caveman.gateway.status.get_running_pid", return_value=12345), \
             patch("caveman.gateway.restart.trigger_restart", side_effect=mock_trigger):
            await handle_restart(ctx)

        assert any("launchd" in r for r in responses)


# ── Transcript Cleaning ──

class TestTranscriptCleaning:
    """Test _clean_transcript_message removes all legacy metadata injections."""

    def test_strip_tool_count_from_assistant(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "assistant", "[使用了 601 个工具调用]\nGateway 已重启"
        )
        assert result == "Gateway 已重启"

    def test_strip_multiple_tool_counts(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "assistant",
            "[使用了 282 个工具调用]\n[使用了 308 个工具调用]\n排查完了"
        )
        assert result == "排查完了"

    def test_strip_format_reminder_from_user(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "user",
            "你好\n[Format: Discord — emoji分隔, **bold**关键词, 空行分段, 禁止表格/标题]"
        )
        assert result == "你好"

    def test_strip_telegram_format_reminder(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "user", "帮我查一下\n[Format: Telegram — emoji, bold, 空行分段, 简洁]"
        )
        assert result == "帮我查一下"

    def test_skip_style_reset_system_message(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "system", "[Style reset] [Format: Discord — emoji分隔...]"
        )
        assert result is None

    def test_keep_normal_system_message(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message("system", "You are Caveman")
        assert result == "You are Caveman"

    def test_strip_compaction_note_from_system(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message(
            "system",
            "You are Caveman\n\n[Note: Earlier turns compacted into a handoff summary.]"
        )
        assert result == "You are Caveman"

    def test_clean_user_without_reminder(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message("user", "正常消息")
        assert result == "正常消息"

    def test_clean_assistant_without_prefix(self):
        from caveman.gateway.runner import _clean_transcript_message
        result = _clean_transcript_message("assistant", "正常回复")
        assert result == "正常回复"
