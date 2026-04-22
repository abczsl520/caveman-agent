"""Tests for MCP OAuth and remaining modules."""
import time
import pytest
from pathlib import Path

from caveman.mcp.oauth import (
    generate_pkce,
    build_auth_url,
    save_tokens,
    load_tokens,
    is_token_expired,
)


class TestPKCE:
    def test_generates_verifier_and_challenge(self):
        verifier, challenge = generate_pkce()
        assert len(verifier) > 40
        assert len(challenge) > 20
        assert verifier != challenge

    def test_unique_each_time(self):
        v1, c1 = generate_pkce()
        v2, c2 = generate_pkce()
        assert v1 != v2


class TestBuildAuthUrl:
    def test_basic_url(self):
        url = build_auth_url(
            "https://auth.example.com/authorize",
            client_id="my-app",
            redirect_uri="http://localhost:8080/callback",
        )
        assert "response_type=code" in url
        assert "client_id=my-app" in url

    def test_with_pkce(self):
        _, challenge = generate_pkce()
        url = build_auth_url(
            "https://auth.example.com/authorize",
            client_id="my-app",
            redirect_uri="http://localhost:8080/callback",
            code_challenge=challenge,
        )
        assert "code_challenge=" in url
        assert "code_challenge_method=S256" in url

    def test_with_scope_and_state(self):
        url = build_auth_url(
            "https://auth.example.com/authorize",
            client_id="my-app",
            redirect_uri="http://localhost:8080/callback",
            scope="read write",
            state="random-state",
        )
        assert "scope=" in url
        assert "state=random-state" in url


class TestTokenStorage:
    def test_save_and_load(self, tmp_path):
        tokens = {"access_token": "abc", "refresh_token": "xyz", "expires_in": 3600}
        save_tokens("test-server", tokens, token_dir=tmp_path)
        loaded = load_tokens("test-server", token_dir=tmp_path)
        assert loaded is not None
        assert loaded["access_token"] == "abc"
        assert "saved_at" in loaded

    def test_load_nonexistent(self, tmp_path):
        assert load_tokens("nonexistent", token_dir=tmp_path) is None


class TestTokenExpiry:
    def test_not_expired(self):
        tokens = {"saved_at": time.time(), "expires_in": 3600}
        assert is_token_expired(tokens) is False

    def test_expired(self):
        tokens = {"saved_at": time.time() - 7200, "expires_in": 3600}
        assert is_token_expired(tokens) is True

    def test_near_expiry(self):
        # Within 60s buffer
        tokens = {"saved_at": time.time() - 3550, "expires_in": 3600}
        assert is_token_expired(tokens) is True
