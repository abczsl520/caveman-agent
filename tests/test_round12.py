"""Tests for Round 12 — REPL + Bridge + Audit + Sandbox + Encryption."""
from __future__ import annotations

import asyncio
import json
import os
import pytest
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ═══════════════════════════════════════════════════════════════
# 1. REPL Interactive Mode
# ═══════════════════════════════════════════════════════════════

class TestREPL:
    """REPL enhancements."""

    def test_tui_imports(self):
        from caveman.cli.tui import (
            interactive_loop, show_banner, show_status,
            show_tool_call, show_tool_result, show_error,
        )
        assert callable(interactive_loop)

    def test_slash_commands_exist(self):
        """All documented /commands are handled via dispatcher."""
        from caveman.commands.dispatcher import dispatch
        assert callable(dispatch)

    @pytest.mark.asyncio
    async def test_handle_exit(self):
        from caveman.commands.dispatcher import dispatch
        agent = MagicMock()
        responses = []
        result = await dispatch("/exit", agent, respond_fn=responses.append)
        assert result == "exit"

    @pytest.mark.asyncio
    async def test_handle_help(self):
        from caveman.commands.dispatcher import dispatch
        agent = MagicMock()
        agent.model = "test-model"
        agent.tool_registry.get_schemas.return_value = []
        agent.memory_manager.total_count = 0
        agent.max_iterations = 50
        agent._tool_call_count = 0
        agent._shield.essence.session_id = "s"
        agent._shield.essence.task = None
        agent._shield.essence.turn_count = 0
        agent._shield.essence.decisions = []
        agent._shield.essence.progress = []
        agent._shield.essence.open_todos = []
        agent._shield.essence.key_data = {}
        agent._reflect.reflections = []
        responses = []
        result = await dispatch("/help", agent, respond_fn=responses.append)
        assert result == "handled"

    @pytest.mark.asyncio
    async def test_handle_status(self):
        from caveman.commands.dispatcher import dispatch
        agent = MagicMock()
        agent.model = "test-model"
        agent.tool_registry.get_schemas.return_value = []
        agent.memory_manager.total_count = 5
        agent.max_iterations = 50
        agent._tool_call_count = 3
        agent._shield.essence.session_id = "s"
        agent._shield.essence.task = "testing"
        agent._shield.essence.turn_count = 1
        agent._shield.essence.decisions = []
        agent._shield.essence.progress = []
        agent._shield.essence.open_todos = []
        agent._shield.essence.key_data = {}
        agent._reflect.reflections = []
        responses = []
        result = await dispatch("/status", agent, respond_fn=responses.append)
        assert result == "handled"

    @pytest.mark.asyncio
    async def test_handle_engines(self):
        from caveman.commands.dispatcher import dispatch
        agent = MagicMock()
        agent.engine_flags.is_enabled.return_value = True
        responses = []
        result = await dispatch("/engines", agent, respond_fn=responses.append)
        assert result == "handled"

    @pytest.mark.asyncio
    async def test_handle_unknown(self):
        from caveman.commands.dispatcher import dispatch
        agent = MagicMock()
        responses = []
        result = await dispatch("/nonexistent", agent, respond_fn=responses.append)
        assert result == "handled"  # Handled (shows error message)


# ═══════════════════════════════════════════════════════════════
# 2. Bridge — 5 Agent Definitions + PTY
# ═══════════════════════════════════════════════════════════════

class TestBridge:
    """CLI agent runner with 5 agents + PTY support."""

    def test_five_agents_defined(self):
        from caveman.bridge.cli_agents import _AGENT_DEFS
        assert "claude" in _AGENT_DEFS
        assert "codex" in _AGENT_DEFS
        assert "gemini" in _AGENT_DEFS
        assert "pi" in _AGENT_DEFS
        assert "aider" in _AGENT_DEFS

    def test_pty_flags(self):
        from caveman.bridge.cli_agents import _AGENT_DEFS
        assert _AGENT_DEFS["claude"]["pty"] is False
        assert _AGENT_DEFS["codex"]["pty"] is True
        assert _AGENT_DEFS["gemini"]["pty"] is True
        assert _AGENT_DEFS["pi"]["pty"] is True
        assert _AGENT_DEFS["aider"]["pty"] is False

    def test_descriptions(self):
        from caveman.bridge.cli_agents import _AGENT_DEFS
        for name, defn in _AGENT_DEFS.items():
            assert "description" in defn, f"{name} missing description"

    @pytest.mark.asyncio
    async def test_run_missing_agent(self):
        from caveman.bridge.cli_agents import CLIAgentRunner, CLIAgentError
        runner = CLIAgentRunner()
        with pytest.raises(CLIAgentError):
            await runner.run("nonexistent", "test")

    @pytest.mark.asyncio
    async def test_run_pipe_not_found(self):
        """Non-PTY agent with missing binary returns exit 127."""
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner(agents={
            "fake": {"cmd": ["__nonexistent_binary__"], "pty": False, "timeout": 5},
        })
        result = await runner.run("fake", "test")
        assert result.exit_code == 127
        assert "not found" in result.output.lower()

    @pytest.mark.asyncio
    async def test_run_pipe_success(self):
        """Pipe-mode agent can run echo."""
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner(agents={
            "echo": {"cmd": ["echo"], "pty": False, "timeout": 5},
        })
        result = await runner.run("echo", "hello world")
        assert result.exit_code == 0
        assert "hello world" in result.output

    @pytest.mark.asyncio
    async def test_run_pty_not_found(self):
        """PTY-mode agent with missing binary returns exit 127."""
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner(agents={
            "fake_pty": {"cmd": ["__nonexistent_pty__"], "pty": True, "timeout": 5},
        })
        result = await runner.run("fake_pty", "test")
        assert result.exit_code == 127

    def test_available_checks_path(self):
        from caveman.bridge.cli_agents import CLIAgentRunner
        runner = CLIAgentRunner(agents={
            "echo_test": {"cmd": ["echo"], "pty": False, "timeout": 5},
            "missing": {"cmd": ["__no_such_binary__"], "pty": False, "timeout": 5},
        })
        avail = runner.available()
        assert "echo_test" in avail
        assert "missing" not in avail


# ═══════════════════════════════════════════════════════════════
# 6. Audit Log
# ═══════════════════════════════════════════════════════════════

class TestAuditLog:
    """Audit log export and query."""

    @pytest.mark.asyncio
    async def test_export_creates_file(self, tmp_path):
        from caveman.audit import export_audit_log
        from caveman.events import EventBus
        bus = EventBus()
        out = await export_audit_log(bus, output_path=str(tmp_path / "audit.jsonl"))
        assert Path(out).exists()

    @pytest.mark.asyncio
    async def test_export_with_events(self, tmp_path):
        from caveman.audit import export_audit_log
        from caveman.events import EventBus, EventType
        from caveman.events_store import EventStore

        # Create store with events
        db_path = tmp_path / "events.db"
        store = EventStore(db_path=str(db_path))
        bus = EventBus()
        bus.on_all(store.handle)

        await bus.emit(EventType.LOOP_START, {"task": "test"}, source="test")
        await bus.emit(EventType.TOOL_CALL, {"name": "bash"}, source="test")

        out = await export_audit_log(
            bus, output_path=str(tmp_path / "audit.jsonl"),
            db_path=str(db_path),
        )
        content = Path(out).read_text()
        lines = [l for l in content.strip().split("\n") if l]
        assert len(lines) >= 2

        # Verify structure
        entry = json.loads(lines[0])
        assert "timestamp" in entry
        assert "type" in entry
        assert "source" in entry

    def test_query_audit_log(self, tmp_path):
        from caveman.audit import query_audit_log

        # Write test log
        log_path = tmp_path / "test.jsonl"
        entries = [
            {"timestamp": 1000, "type": "loop_start", "source": "loop", "data": {}},
            {"timestamp": 1001, "type": "tool_call", "source": "tool", "data": {"name": "bash"}},
            {"timestamp": 1002, "type": "loop_end", "source": "loop", "data": {}},
        ]
        log_path.write_text("\n".join(json.dumps(e) for e in entries))

        # Query all
        results = query_audit_log(log_path)
        assert len(results) == 3

        # Filter by type
        results = query_audit_log(log_path, event_type="tool_call")
        assert len(results) == 1

        # Filter by source
        results = query_audit_log(log_path, source="loop")
        assert len(results) == 2


# ═══════════════════════════════════════════════════════════════
# 7a. Sandbox Execution
# ═══════════════════════════════════════════════════════════════

class TestSandbox:
    """Sandboxed code execution."""

    @pytest.mark.asyncio
    async def test_python_execution(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute("print('hello from sandbox')", language="python")
        assert result.exit_code == 0
        assert "hello from sandbox" in result.stdout

    @pytest.mark.asyncio
    async def test_bash_blocked_by_default(self):
        """Bash is removed from default allowed commands for security."""
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute("echo 'bash works'", language="bash")
        assert result.exit_code == 1
        assert "Unsupported" in result.stderr

    @pytest.mark.asyncio
    async def test_bash_allowed_when_configured(self):
        """Bash can be explicitly allowed via config."""
        from caveman.security.sandbox import Sandbox, SandboxConfig
        sb = Sandbox(SandboxConfig(allowed_commands=["python3", "python", "node", "bash"]))
        result = await sb.execute("echo 'bash works'", language="bash")
        assert result.exit_code == 0
        assert "bash works" in result.stdout

    @pytest.mark.asyncio
    async def test_timeout(self):
        from caveman.security.sandbox import Sandbox, SandboxConfig
        sb = Sandbox(SandboxConfig(timeout=2))
        result = await sb.execute("import time; time.sleep(10)", language="python")
        assert result.timed_out is True

    @pytest.mark.asyncio
    async def test_error_capture(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute("raise ValueError('test error')", language="python")
        assert result.exit_code != 0
        assert "ValueError" in result.stderr

    @pytest.mark.asyncio
    async def test_restricted_env(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute(
            "import os; print(os.environ.get('CAVEMAN_SANDBOX', 'missing'))",
            language="python",
        )
        assert "1" in result.stdout

    @pytest.mark.asyncio
    async def test_additional_files(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute(
            "with open('data.txt') as f: print(f.read())",
            language="python",
            files={"data.txt": "hello data"},
        )
        assert "hello data" in result.stdout

    @pytest.mark.asyncio
    async def test_validate_skill(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        valid, msg = await sb.validate_skill("def hello(): return 42")
        assert valid is True

    @pytest.mark.asyncio
    async def test_validate_invalid_skill(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        valid, msg = await sb.validate_skill("def hello(: return")
        assert valid is False

    @pytest.mark.asyncio
    async def test_unsupported_language(self):
        from caveman.security.sandbox import Sandbox
        sb = Sandbox()
        result = await sb.execute("code", language="cobol")
        assert result.exit_code == 1
        assert "Unsupported" in result.stderr


# ═══════════════════════════════════════════════════════════════
# 7b. E2E Encryption
# ═══════════════════════════════════════════════════════════════

class TestEncryption:
    """AES-256-GCM encryption."""

    def test_encrypt_decrypt_roundtrip(self):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("test-passphrase")
        plaintext = b"Hello, Caveman! This is secret data."
        blob = enc.encrypt(plaintext)
        decrypted = enc.decrypt(blob)
        assert decrypted == plaintext

    def test_text_roundtrip(self):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("my-secret-key")
        original = "Server IP: 203.0.113.10, API key: sk-test-123"
        encrypted = enc.encrypt_text(original)
        assert encrypted != original  # Should be different
        decrypted = enc.decrypt_text(encrypted)
        assert decrypted == original

    def test_different_passphrases_fail(self):
        from caveman.security.encryption import Encryptor
        enc1 = Encryptor("passphrase-1")
        enc2 = Encryptor("passphrase-2")
        blob = enc1.encrypt(b"secret")
        with pytest.raises((ValueError, Exception)):
            enc2.decrypt(blob)

    def test_tamper_detection(self):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("test")
        blob = enc.encrypt(b"important data")
        # Tamper with ciphertext
        tampered = bytearray(blob.ciphertext)
        if tampered:
            tampered[0] ^= 0xFF
        blob.ciphertext = bytes(tampered)
        with pytest.raises((ValueError, Exception)):
            enc.decrypt(blob)

    def test_blob_serialization(self):
        from caveman.security.encryption import EncryptedBlob
        blob = EncryptedBlob(
            ciphertext=b"ct", nonce=b"n" * 12, salt=b"s" * 32, tag=b"t" * 16,
        )
        serialized = blob.to_bytes()
        restored = EncryptedBlob.from_bytes(serialized)
        assert restored.ciphertext == blob.ciphertext
        assert restored.nonce == blob.nonce
        assert restored.salt == blob.salt
        assert restored.tag == blob.tag

    def test_base64_roundtrip(self):
        from caveman.security.encryption import EncryptedBlob
        blob = EncryptedBlob(
            ciphertext=b"data", nonce=b"n" * 12, salt=b"s" * 32, tag=b"t" * 16,
        )
        b64 = blob.to_base64()
        restored = EncryptedBlob.from_base64(b64)
        assert restored.ciphertext == blob.ciphertext

    def test_file_encrypt_decrypt(self, tmp_path):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("file-key")

        # Create test file
        original = tmp_path / "secret.txt"
        original.write_text("top secret content")

        # Encrypt
        enc_path = enc.encrypt_file(original)
        assert enc_path.suffix == ".enc"
        assert enc_path.read_bytes() != b"top secret content"

        # Decrypt
        dec_path = enc.decrypt_file(enc_path)
        assert dec_path.read_text() == "top secret content"

    def test_large_data(self):
        """Encrypt/decrypt 1MB of data."""
        from caveman.security.encryption import Encryptor
        enc = Encryptor("large-data-key")
        data = os.urandom(1_000_000)
        blob = enc.encrypt(data)
        decrypted = enc.decrypt(blob)
        assert decrypted == data

    def test_empty_data(self):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("empty")
        blob = enc.encrypt(b"")
        assert enc.decrypt(blob) == b""

    def test_unicode_passphrase(self):
        from caveman.security.encryption import Encryptor
        enc = Encryptor("密码🔑パスワード")
        blob = enc.encrypt(b"unicode key test")
        assert enc.decrypt(blob) == b"unicode key test"
