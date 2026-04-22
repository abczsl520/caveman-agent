"""Tests for P10: redaction, message splitting, rate limiter, attachments, context pruning, vision, LLM router."""
from __future__ import annotations

import time
from pathlib import Path

import pytest


# ── Redaction ──

class TestRedaction:
    def test_redact_openai_key(self):
        from caveman.gateway.redaction import redact_secrets
        text = "My key is sk-abc123456789012345678901234567890123"
        assert "[REDACTED]" in redact_secrets(text)

    def test_redact_github_token(self):
        from caveman.gateway.redaction import redact_secrets
        text = "Token: ghp_abcdefghijklmnopqrstuvwxyz1234567890"
        assert "[REDACTED]" in redact_secrets(text)

    def test_redact_pii_email(self):
        from caveman.gateway.redaction import redact_pii
        text = "Contact me at user@example.com"
        assert "[PII]" in redact_pii(text)

    def test_redact_pii_phone(self):
        from caveman.gateway.redaction import redact_pii
        text = "Call +1-555-123-4567"
        assert "[PII]" in redact_pii(text)

    def test_detect_secrets(self):
        from caveman.gateway.redaction import detect_secrets
        text = "key: sk-abc12345678901234567890"
        findings = detect_secrets(text)
        assert len(findings) >= 1

    def test_safe_text(self):
        from caveman.gateway.redaction import is_safe_for_external
        assert is_safe_for_external("Hello world")
        assert not is_safe_for_external("key: sk-abc12345678901234567890")

    def test_redact_all(self):
        from caveman.gateway.redaction import redact_all
        text = "Key: sk-abc12345678901234567890, email: test@test.com"
        result = redact_all(text)
        assert "sk-" not in result
        assert "test@" not in result


# ── Message Splitting ──

class TestMessageSplitting:
    def test_short_message(self):
        from caveman.gateway.message_splitting import split_message
        assert split_message("hello") == ["hello"]

    def test_long_message_discord(self):
        from caveman.gateway.message_splitting import split_message
        text = "A" * 5000
        chunks = split_message(text, "discord")
        assert len(chunks) >= 3
        assert all(len(c) <= 2000 for c in chunks)

    def test_preserves_code_blocks(self):
        from caveman.gateway.message_splitting import split_message
        text = "Before\n\n```python\n" + "x = 1\n" * 100 + "```\n\nAfter"
        chunks = split_message(text, "discord")
        # Code block should not be split mid-block
        for chunk in chunks:
            opens = chunk.count("```")
            assert opens % 2 == 0 or chunk.strip().endswith("```")

    def test_platform_limits(self):
        from caveman.gateway.message_splitting import PLATFORM_LIMITS
        assert PLATFORM_LIMITS["discord"] == 2000
        assert PLATFORM_LIMITS["telegram"] == 4096

    def test_estimate_count(self):
        from caveman.gateway.message_splitting import estimate_message_count
        assert estimate_message_count("short") == 1
        assert estimate_message_count("A" * 5000, "discord") >= 3


# ── Rate Limiter ──

class TestRateLimiter:
    def test_token_bucket(self):
        from caveman.gateway.rate_limiter import TokenBucket
        bucket = TokenBucket(capacity=5, refill_rate=1.0)
        assert bucket.try_consume(3)
        assert bucket.try_consume(2)
        assert not bucket.try_consume(1)

    def test_wait_time(self):
        from caveman.gateway.rate_limiter import TokenBucket
        bucket = TokenBucket(capacity=1, refill_rate=1.0)
        bucket.try_consume(1)
        wait = bucket.wait_time(1)
        assert wait > 0

    def test_rate_limiter_multi_key(self):
        from caveman.gateway.rate_limiter import RateLimiter
        limiter = RateLimiter(default_capacity=2, default_rate=0.1)
        assert limiter.try_acquire("key1")
        assert limiter.try_acquire("key1")
        assert not limiter.try_acquire("key1")
        # Different key should work
        assert limiter.try_acquire("key2")

    def test_configure(self):
        from caveman.gateway.rate_limiter import RateLimiter
        limiter = RateLimiter()
        limiter.configure("api", capacity=100, rate=10)
        state = limiter.check("api")
        assert state["capacity"] == 100

    def test_reset(self):
        from caveman.gateway.rate_limiter import RateLimiter
        limiter = RateLimiter(default_capacity=1)
        limiter.try_acquire("k1")
        limiter.reset("k1")
        assert limiter.try_acquire("k1")


# ── Attachments ──

class TestAttachments:
    def test_attachment_types(self):
        from caveman.gateway.attachments import Attachment
        img = Attachment(filename="photo.png", mime_type="image/png")
        assert img.is_image
        assert not img.is_audio

    def test_validate_size(self, tmp_path):
        from caveman.gateway.attachments import AttachmentHandler, Attachment
        handler = AttachmentHandler(storage_dir=tmp_path)
        small = Attachment(filename="ok.txt", size=100, mime_type="text/plain")
        assert handler.validate(small)["valid"]
        big = Attachment(filename="huge.bin", size=100 * 1024 * 1024, mime_type="text/plain")
        assert not handler.validate(big, "discord")["valid"]

    def test_list_empty(self, tmp_path):
        from caveman.gateway.attachments import AttachmentHandler
        handler = AttachmentHandler(storage_dir=tmp_path)
        assert handler.list_attachments() == []


# ── Context Pruning ──

class TestContextPruning:
    def test_no_pruning_needed(self):
        from caveman.gateway.context_pruning import ContextPruner
        pruner = ContextPruner()
        messages = [{"role": "user", "content": "hi"}]
        result = pruner.prune(messages, current_tokens=100)
        assert len(result) == 1

    def test_fifo_pruning(self):
        from caveman.gateway.context_pruning import ContextPruner, PruningConfig
        config = PruningConfig(max_tokens=100, target_ratio=0.5, strategy="fifo", preserve_recent=2)
        pruner = ContextPruner(config)
        messages = [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "old1"},
            {"role": "assistant", "content": "old2"},
            {"role": "user", "content": "recent1"},
            {"role": "assistant", "content": "recent2"},
        ]
        result = pruner.prune(messages, current_tokens=200)
        assert len(result) <= 4  # system + 2 recent + maybe 1

    def test_context_cache(self):
        from caveman.gateway.context_pruning import ContextCache
        cache = ContextCache(ttl=3600)
        cache.set("k1", [{"role": "user", "content": "hi"}])
        assert cache.get("k1") is not None
        cache.invalidate("k1")
        assert cache.get("k1") is None

    def test_cache_key(self):
        from caveman.gateway.context_pruning import ContextCache
        k1 = ContextCache.cache_key("session1", 10)
        k2 = ContextCache.cache_key("session1", 11)
        assert k1 != k2


# ── Vision Tools ──

class TestVisionTools:
    def test_build_vision_message(self):
        from caveman.tools.builtin.vision_tools import build_vision_message
        msg = build_vision_message("What is this?", ["https://example.com/img.png"])
        assert msg["role"] == "user"
        assert len(msg["content"]) == 2

    def test_extract_image_urls(self):
        from caveman.tools.builtin.vision_tools import extract_image_urls
        text = "Look at ![photo](https://example.com/photo.png) and https://other.com/img.jpg"
        urls = extract_image_urls(text)
        assert len(urls) == 2

    def test_is_image_size_error(self):
        from caveman.tools.builtin.vision_tools import is_image_size_error
        assert is_image_size_error(Exception("image too large"))
        assert is_image_size_error(Exception("413 Payload Too Large"))
        assert not is_image_size_error(Exception("invalid model"))


# ── LLM Router ──

class TestLLMRouter:
    def test_resolve_provider(self):
        from caveman.tools.builtin.llm_router import resolve_provider
        config = resolve_provider("claude-opus-4-6")
        assert config is not None
        assert config.provider == "anthropic"

    def test_resolve_openai(self):
        from caveman.tools.builtin.llm_router import resolve_provider
        config = resolve_provider("gpt-4o")
        assert config is not None
        assert config.provider == "openai"

    def test_resolve_unknown(self):
        from caveman.tools.builtin.llm_router import resolve_provider
        assert resolve_provider("unknown-model") is None

    def test_llm_response(self):
        from caveman.tools.builtin.llm_router import LLMResponse
        resp = LLMResponse(text="hello", model="test", provider="test")
        assert resp.text == "hello"
