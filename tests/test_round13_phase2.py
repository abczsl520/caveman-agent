"""Tests for Round 13 Phase 2 — redaction + context references."""
from __future__ import annotations

import os
import pytest
from pathlib import Path

from caveman.security.redact import redact_text as redact, _mask, RedactingFormatter
from caveman.agent.context_refs import (
    parse_refs, expand_refs, _is_sensitive, ContextRef,
)


# --- Redaction ---

class TestRedact:
    def test_openai_key(self):
        text = "key is sk-abc123def456ghi789jkl012mno345"
        result = redact(text)
        assert "sk-abc123def456ghi789jkl012mno345" not in result
        assert "..." in result  # masked with prefix...suffix

    def test_github_pat(self):
        result = redact("token: ghp_1234567890abcdefghij")
        assert "ghp_1234567890abcdefghij" not in result

    def test_aws_key(self):
        result = redact("AKIAIOSFODNN7EXAMPLE")
        # AWS keys are 20 chars, masked to ***
        assert "AKIAIOSFODNN7EXAMPLE" not in result

    def test_env_assignment(self):
        result = redact("OPENAI_API_KEY=sk-verylongkeyvalue1234567890")
        assert "***" in result
        assert "OPENAI_API_KEY=" in result

    def test_json_field(self):
        result = redact('"api_key": "sk-verylongkeyvalue1234567890"')
        assert "***" in result

    def test_auth_header(self):
        result = redact("Authorization: Bearer sk-verylongkeyvalue1234567890")
        assert "***" in result

    def test_telegram_token(self):
        result = redact("bot123456789:ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefgh")
        assert "***" in result

    def test_private_key(self):
        key = "-----BEGIN RSA PRIVATE KEY-----\nMIIE...\n-----END RSA PRIVATE KEY-----"
        result = redact(key)
        assert "REDACTED" in result

    def test_db_connection(self):
        result = redact("postgres://user:secretpass@localhost:5432/db")
        assert "secretpass" not in result
        assert "***" in result

    def test_phone_number(self):
        result = redact("call me at +8613812345678")
        assert "****" in result
        assert "+8613812345678" not in result

    def test_no_false_positive(self):
        text = "Hello world, this is normal text."
        assert redact(text) == text

    def test_empty(self):
        assert redact("") == ""
        assert redact(None) == ""

    def test_mask_short(self):
        assert _mask("short") == "***"

    def test_mask_long(self):
        result = _mask("sk-abc123def456ghi789jkl012")
        assert result.startswith("sk-abc")
        assert result.endswith("l012")
        assert "..." in result

    def test_tavily_key(self):
        result = redact("tvly-abc123def456ghi789jkl012")
        assert "tvly-abc123def456ghi789jkl012" not in result

    def test_custom_key(self):
        result = redact("cr_87718e41a566ca65f3cba501b9f67ebc98994868d32918cb")
        assert "cr_87718e41a566ca65f3cba501b9f67ebc98994868d32918cb" not in result

    def test_multiple_secrets(self):
        text = "key1=sk-abc123def456ghi789jkl012 key2=ghp_1234567890abcdefghij"
        result = redact(text)
        assert result.count("***") >= 2 or result.count("...") >= 2

    def test_groq_key(self):
        result = redact("gsk_abc123def456ghi789jkl012")
        assert "gsk_abc123def456ghi789jkl012" not in result

    def test_formatter(self):
        import logging
        fmt = RedactingFormatter("%(message)s")
        record = logging.LogRecord(
            "test", logging.INFO, "", 0,
            "key is sk-abc123def456ghi789jkl012mno345", (), None,
        )
        result = fmt.format(record)
        assert "***" in result or "..." in result


# --- Context References ---

class TestParseRefs:
    def test_file_ref(self):
        refs = parse_refs("check @file:src/main.py please")
        assert len(refs) == 1
        assert refs[0].kind == "file"
        assert refs[0].target == "src/main.py"

    def test_file_with_lines(self):
        refs = parse_refs("see @file:main.py:10-20")
        assert len(refs) == 1
        assert refs[0].line_start == 10
        assert refs[0].line_end == 20

    def test_file_single_line(self):
        refs = parse_refs("see @file:main.py:42")
        assert len(refs) == 1
        assert refs[0].line_start == 42
        assert refs[0].line_end == 42

    def test_quoted_file(self):
        refs = parse_refs('check @file:`src/my file.py`')
        assert len(refs) == 1
        assert refs[0].target == "src/my file.py"

    def test_folder_ref(self):
        refs = parse_refs("list @folder:src/")
        assert len(refs) == 1
        assert refs[0].kind == "folder"

    def test_url_ref(self):
        refs = parse_refs("fetch @url:https://example.com")
        assert len(refs) == 1
        assert refs[0].kind == "url"
        assert refs[0].target == "https://example.com"

    def test_diff_ref(self):
        refs = parse_refs("show me @diff")
        assert len(refs) == 1
        assert refs[0].kind == "diff"

    def test_staged_ref(self):
        refs = parse_refs("review @staged")
        assert len(refs) == 1
        assert refs[0].kind == "staged"

    def test_multiple_refs(self):
        refs = parse_refs("compare @file:a.py and @file:b.py")
        assert len(refs) == 2

    def test_no_refs(self):
        assert parse_refs("no references here") == []

    def test_email_not_matched(self):
        # @ in email should not match
        refs = parse_refs("email user@example.com")
        assert len(refs) == 0


class TestExpandRefs:
    def test_file_expansion(self, tmp_path):
        f = tmp_path / "test.py"
        f.write_text("print('hello')\n")
        result = expand_refs(
            "check @file:test.py",
            cwd=str(tmp_path),
        )
        assert "print('hello')" in result.message
        assert result.tokens_added > 0

    def test_file_not_found(self, tmp_path):
        result = expand_refs(
            "check @file:missing.py",
            cwd=str(tmp_path),
        )
        assert len(result.warnings) == 1
        assert "not found" in result.warnings[0]

    def test_folder_expansion(self, tmp_path):
        (tmp_path / "a.py").write_text("a")
        (tmp_path / "b.py").write_text("b")
        result = expand_refs(
            "list @folder:.",
            cwd=str(tmp_path),
        )
        assert "a.py" in result.message
        assert "b.py" in result.message

    def test_sensitive_file_blocked(self, tmp_path):
        ssh_dir = Path.home() / ".ssh"
        if ssh_dir.exists():
            result = expand_refs(
                f"read @file:{ssh_dir / 'id_rsa'}",
                cwd=str(tmp_path),
            )
            assert result.blocked or len(result.warnings) > 0

    def test_token_budget(self, tmp_path):
        f = tmp_path / "big.txt"
        f.write_text("x" * 100_000)
        result = expand_refs(
            "read @file:big.txt",
            cwd=str(tmp_path),
            token_budget=1000,
        )
        assert "truncated" in result.message or len(result.warnings) > 0

    def test_no_refs_passthrough(self):
        result = expand_refs("no refs here")
        assert result.message == "no refs here"
        assert result.tokens_added == 0

    def test_line_range(self, tmp_path):
        f = tmp_path / "lines.py"
        f.write_text("line1\nline2\nline3\nline4\nline5\n")
        result = expand_refs(
            "check @file:lines.py:2-4",
            cwd=str(tmp_path),
        )
        assert "line2" in result.message
        assert "line4" in result.message

    def test_url_skipped_sync(self):
        result = expand_refs("fetch @url:https://example.com")
        assert len(result.warnings) == 1
        assert "URL" in result.warnings[0]


class TestSensitiveCheck:
    def test_ssh_dir(self):
        home = Path.home()
        assert _is_sensitive(home / ".ssh" / "id_rsa", home)

    def test_aws_dir(self):
        home = Path.home()
        assert _is_sensitive(home / ".aws" / "credentials", home)

    def test_normal_file(self):
        home = Path.home()
        assert not _is_sensitive(home / "projects" / "main.py", home)

    def test_outside_home(self):
        assert not _is_sensitive(Path("/tmp/test.py"), Path.home())
