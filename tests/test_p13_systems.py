"""Tests for P13: context_engine, secrets, security_audit, daemon, media_understanding."""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import pytest


# ── Context Engine ──

class TestContextEngine:
    def test_default_engine_ingest(self):
        from caveman.agent.context_engine import DefaultContextEngine
        engine = DefaultContextEngine(model="gpt-4o")
        result = engine.ingest("s1", {"role": "user", "content": "hello"})
        assert result.ingested

    def test_default_engine_assemble(self):
        from caveman.agent.context_engine import DefaultContextEngine
        engine = DefaultContextEngine(model="gpt-4o")
        messages = [{"role": "user", "content": "hi"}]
        result = engine.assemble("s1", messages, token_budget=128000)
        assert len(result.messages) == 1

    def test_default_engine_compact_below_threshold(self):
        from caveman.agent.context_engine import DefaultContextEngine
        engine = DefaultContextEngine(model="gpt-4o")
        engine.ingest("s1", {"role": "user", "content": "hi"})
        result = engine.compact("s1")
        assert not result.compacted

    def test_engine_registry(self):
        from caveman.agent.context_engine import (
            DefaultContextEngine, register_engine, get_engine, list_engines,
        )
        engine = DefaultContextEngine(model="gpt-4o")
        register_engine(engine)
        assert get_engine("default") is engine
        infos = list_engines()
        assert any(i.id == "default" for i in infos)

    def test_engine_info(self):
        from caveman.agent.context_engine import DefaultContextEngine
        engine = DefaultContextEngine()
        assert engine.info.name == "DefaultContextEngine"
        assert engine.info.owns_compaction


# ── Secrets ──

class TestSecrets:
    def test_set_and_get(self, tmp_path):
        from caveman.gateway.secrets import SecretsManager
        import caveman.gateway.secrets as mod
        old = mod._CREDENTIAL_FILE
        mod._CREDENTIAL_FILE = tmp_path / "creds.json"
        try:
            mgr = SecretsManager(secrets_dir=tmp_path)
            mgr.set("TEST_KEY", "test_value", provider="openai")
            assert mgr.get("TEST_KEY") == "test_value"
        finally:
            mod._CREDENTIAL_FILE = old

    def test_get_from_env(self, tmp_path):
        from caveman.gateway.secrets import SecretsManager
        import caveman.gateway.secrets as mod
        old = mod._CREDENTIAL_FILE
        mod._CREDENTIAL_FILE = tmp_path / "creds.json"
        try:
            mgr = SecretsManager(secrets_dir=tmp_path)
            os.environ["_TEST_SECRET_XYZ"] = "env_value"
            assert mgr.get("_TEST_SECRET_XYZ") == "env_value"
        finally:
            mod._CREDENTIAL_FILE = old
            os.environ.pop("_TEST_SECRET_XYZ", None)

    def test_delete(self, tmp_path):
        from caveman.gateway.secrets import SecretsManager
        import caveman.gateway.secrets as mod
        old = mod._CREDENTIAL_FILE
        mod._CREDENTIAL_FILE = tmp_path / "creds.json"
        try:
            mgr = SecretsManager(secrets_dir=tmp_path)
            mgr.set("K", "V")
            assert mgr.delete("K")
            assert mgr.get("K") is None
        finally:
            mod._CREDENTIAL_FILE = old

    def test_rotate(self, tmp_path):
        from caveman.gateway.secrets import SecretsManager
        import caveman.gateway.secrets as mod
        old = mod._CREDENTIAL_FILE
        mod._CREDENTIAL_FILE = tmp_path / "creds.json"
        try:
            mgr = SecretsManager(secrets_dir=tmp_path)
            mgr.set("K", "old")
            mgr.rotate("K", "new")
            assert mgr.get("K") == "new"
        finally:
            mod._CREDENTIAL_FILE = old

    def test_audit(self, tmp_path):
        from caveman.gateway.secrets import SecretsManager
        import caveman.gateway.secrets as mod
        old = mod._CREDENTIAL_FILE
        mod._CREDENTIAL_FILE = tmp_path / "creds.json"
        try:
            mgr = SecretsManager(secrets_dir=tmp_path)
            mgr.set("K1", "V1")
            mgr.set("K2", "V2", expires_at=time.time() - 100)
            audit = mgr.audit()
            assert audit["total"] == 2
            assert audit["expired"] == 1
        finally:
            mod._CREDENTIAL_FILE = old


# ── Security Audit ──

class TestSecurityAudit:
    def test_run_audit(self, tmp_path):
        from caveman.gateway.security_audit import run_audit
        report = run_audit(home_dir=tmp_path)
        assert report.score >= 0
        assert report.scanned_at > 0

    def test_overly_permissive(self, tmp_path):
        from caveman.gateway.security_audit import _check_file_permissions
        secrets_dir = tmp_path / "secrets"
        secrets_dir.mkdir()
        cred_file = secrets_dir / "credentials.json"
        cred_file.write_text("{}")
        cred_file.chmod(0o644)  # Too permissive
        findings = _check_file_permissions(tmp_path)
        assert len(findings) >= 1
        assert findings[0].severity == "high"

    def test_score_computation(self):
        from caveman.gateway.security_audit import AuditReport, AuditFinding
        report = AuditReport(findings=[
            AuditFinding(category="test", severity="critical", title="bad"),
            AuditFinding(category="test", severity="high", title="bad2"),
        ])
        score = report.compute_score()
        assert score < 100


# ── Daemon ──

class TestDaemon:
    def test_get_status_not_running(self, tmp_path):
        from caveman.gateway.daemon import get_status
        import caveman.gateway.daemon as mod
        old = mod._PID_FILE
        mod._PID_FILE = tmp_path / "daemon.pid"
        try:
            status = get_status()
            assert not status.running
        finally:
            mod._PID_FILE = old

    def test_get_status_stale_pid(self, tmp_path):
        from caveman.gateway.daemon import get_status
        import caveman.gateway.daemon as mod
        old = mod._PID_FILE
        mod._PID_FILE = tmp_path / "daemon.pid"
        try:
            (tmp_path / "daemon.pid").write_text("99999999")
            status = get_status()
            assert not status.running
        finally:
            mod._PID_FILE = old


# ── Media Understanding ──

class TestMediaUnderstanding:
    def test_detect_media_type(self):
        from caveman.tools.builtin.media_understanding import detect_media_type
        assert detect_media_type("photo.png") == "image"
        assert detect_media_type("song.mp3") == "audio"
        assert detect_media_type("movie.mp4") == "video"
        assert detect_media_type("doc.pdf") == "document"
        assert detect_media_type("file.xyz") == "unknown"

    def test_analyze_document_text(self, tmp_path):
        from caveman.tools.builtin.media_understanding import analyze_document
        doc = tmp_path / "test.txt"
        doc.write_text("Hello world content")
        result = analyze_document(str(doc))
        assert result.text_content == "Hello world content"
        assert result.media_type == "document"

    def test_analyze_document_missing(self):
        from caveman.tools.builtin.media_understanding import analyze_document
        result = analyze_document("/nonexistent.txt")
        assert result.error

    def test_analyze_image_missing(self):
        from caveman.tools.builtin.media_understanding import analyze_image
        result = analyze_image("/nonexistent.png")
        assert result.error
