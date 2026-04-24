"""Tests for proxy identity injection defense."""
import pytest
from caveman.providers.anthropic_adapter import (
    _maybe_prepend_identity_anchor,
    build_api_kwargs,
)


class TestIdentityAnchor:
    """Test _maybe_prepend_identity_anchor."""

    def test_no_proxy_no_anchor(self):
        """Direct Anthropic API — no anchor prepended."""
        blocks = [{"type": "text", "text": "You are Caveman."}]
        result = _maybe_prepend_identity_anchor(blocks, base_url=None)
        assert result == blocks

    def test_anthropic_url_no_anchor(self):
        """Anthropic official URL — no anchor."""
        blocks = [{"type": "text", "text": "You are Caveman."}]
        result = _maybe_prepend_identity_anchor(blocks, base_url="https://api.anthropic.com")
        assert result == blocks

    def test_proxy_url_adds_anchor(self):
        """Non-Anthropic URL — anchor prepended."""
        blocks = [{"type": "text", "text": "You are Caveman."}]
        result = _maybe_prepend_identity_anchor(blocks, base_url="http://198.51.100.20:4200")
        assert len(result) == 2
        assert "IDENTITY" in result[0]["text"]
        assert result[0]["cache_control"] == {"type": "ephemeral"}
        assert result[1] == blocks[0]

    def test_build_api_kwargs_with_proxy(self):
        """build_api_kwargs passes base_url to identity anchor."""
        kwargs = build_api_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            system="You are Caveman.",
            base_url="http://proxy.example.com:4200",
        )
        system_blocks = kwargs.get("system", [])
        # Should have anchor + original system
        assert len(system_blocks) >= 2
        assert "IDENTITY" in system_blocks[0]["text"]

    def test_build_api_kwargs_without_proxy(self):
        """build_api_kwargs without base_url — no anchor."""
        kwargs = build_api_kwargs(
            model="claude-opus-4-6",
            messages=[{"role": "user", "content": "hi"}],
            system="You are Caveman.",
        )
        system_blocks = kwargs.get("system", [])
        # No anchor — just the original system
        assert len(system_blocks) == 1
        assert "You are Caveman." in system_blocks[0]["text"]
