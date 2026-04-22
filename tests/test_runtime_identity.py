"""Tests for runtime identity and environment hygiene."""
import os

import pytest

from caveman.runtime_identity import (
    CAVEMAN_SERVICE_NAME,
    CAVEMAN_VERSION,
    build_clean_env,
    get_runtime_identity,
    sanitize_environment,
)


class TestSanitizeEnvironment:
    def test_removes_openclaw_vars(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_GATEWAY_PORT", "18789")
        monkeypatch.setenv("OPENCLAW_SERVICE_KIND", "gateway")
        monkeypatch.setenv("OPENCLAW_SHELL", "exec")

        removed = sanitize_environment()

        assert "OPENCLAW_GATEWAY_PORT" in removed
        assert "OPENCLAW_SERVICE_KIND" in removed
        assert "OPENCLAW_SHELL" in removed
        assert "OPENCLAW_GATEWAY_PORT" not in os.environ
        assert "OPENCLAW_SERVICE_KIND" not in os.environ

    def test_removes_hermes_vars(self, monkeypatch):
        monkeypatch.setenv("HERMES_API_KEY", "test")
        monkeypatch.setenv("HERMES_MODEL", "llama")

        removed = sanitize_environment()

        assert "HERMES_API_KEY" in removed
        assert "HERMES_MODEL" in removed

    def test_preserves_caveman_vars(self, monkeypatch):
        monkeypatch.setenv("CAVEMAN_HOME", "/tmp/test")

        sanitize_environment()

        assert os.environ.get("CAVEMAN_HOME") == "/tmp/test"

    def test_preserves_normal_vars(self, monkeypatch):
        monkeypatch.setenv("PATH", "/usr/bin")
        monkeypatch.setenv("HOME", "/home/test")

        sanitize_environment()

        assert "PATH" in os.environ
        assert "HOME" in os.environ

    def test_sets_caveman_identity(self):
        sanitize_environment()

        assert os.environ.get("CAVEMAN_SERVICE_NAME") == CAVEMAN_SERVICE_NAME
        assert os.environ.get("CAVEMAN_VERSION") == CAVEMAN_VERSION
        assert os.environ.get("CAVEMAN_PID") == str(os.getpid())

    def test_idempotent(self, monkeypatch):
        monkeypatch.setenv("OPENCLAW_TEST", "value")

        sanitize_environment()
        removed2 = sanitize_environment()

        # Second call should find nothing to remove
        assert len(removed2) == 0


class TestBuildCleanEnv:
    def test_has_term_and_no_color(self):
        env = build_clean_env()
        assert env["TERM"] == "dumb"
        assert env["NO_COLOR"] == "1"

    def test_no_foreign_vars(self, monkeypatch):
        # First sanitize
        monkeypatch.setenv("OPENCLAW_LEAK", "should_not_appear")
        sanitize_environment()

        env = build_clean_env()
        assert "OPENCLAW_LEAK" not in env

    def test_extra_vars_merged(self):
        env = build_clean_env(extra={"MY_VAR": "hello"})
        assert env["MY_VAR"] == "hello"

    def test_has_caveman_identity(self):
        sanitize_environment()
        env = build_clean_env()
        assert env.get("CAVEMAN_SERVICE_NAME") == CAVEMAN_SERVICE_NAME


class TestRuntimeIdentity:
    def test_returns_dict(self):
        sanitize_environment()
        identity = get_runtime_identity()
        assert identity["service"] == "caveman"
        assert identity["version"] == CAVEMAN_VERSION
        assert identity["pid"] == str(os.getpid())
