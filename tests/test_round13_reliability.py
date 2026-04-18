"""Tests for Round 13 — reliability chassis."""
from __future__ import annotations

import asyncio
import time
import pytest

from caveman.providers.error_classifier import (
    ClassifiedError,
    FailoverReason,
    classify_error,
)
from caveman.providers.retry import jittered_backoff, retry_with_backoff
from caveman.providers.credential_pool import (
    Credential,
    CredentialPool,
    RotationStrategy,
    COOLDOWN_SECONDS,
)


# --- Error Classifier ---

class _FakeError(Exception):
    def __init__(self, msg="error", status_code=None, body=None):
        super().__init__(msg)
        if status_code is not None:
            self.status_code = status_code
        if body is not None:
            self.body = body


class TestErrorClassifier:
    def test_rate_limit_429(self):
        e = _FakeError("too many requests", status_code=429)
        c = classify_error(e)
        assert c.reason == FailoverReason.rate_limit
        assert c.retryable
        assert c.should_rotate

    def test_auth_401(self):
        c = classify_error(_FakeError("unauthorized", status_code=401))
        assert c.reason == FailoverReason.auth
        assert not c.retryable
        assert c.should_rotate

    def test_billing_402(self):
        c = classify_error(_FakeError("insufficient credits", status_code=402))
        assert c.reason == FailoverReason.billing
        assert not c.retryable
        assert c.should_rotate

    def test_transient_402(self):
        c = classify_error(_FakeError("usage limit, try again in 5 minutes", status_code=402))
        assert c.reason == FailoverReason.rate_limit
        assert c.retryable

    def test_context_overflow_400(self):
        c = classify_error(_FakeError("context length exceeded", status_code=400))
        assert c.reason == FailoverReason.context_overflow
        assert c.should_compress

    def test_context_overflow_chinese(self):
        c = classify_error(_FakeError("超过最大长度"))
        assert c.reason == FailoverReason.context_overflow

    def test_model_not_found_404(self):
        c = classify_error(_FakeError("not found", status_code=404))
        assert c.reason == FailoverReason.model_not_found
        assert not c.retryable
        assert c.should_fallback

    def test_server_error_500(self):
        c = classify_error(_FakeError("internal error", status_code=500))
        assert c.reason == FailoverReason.server_error
        assert c.retryable

    def test_overloaded_503(self):
        c = classify_error(_FakeError("overloaded", status_code=503))
        assert c.reason == FailoverReason.overloaded
        assert c.retryable

    def test_payload_too_large_413(self):
        c = classify_error(_FakeError("too large", status_code=413))
        assert c.reason == FailoverReason.payload_too_large
        assert c.should_compress

    def test_timeout_transport(self):
        class APITimeoutError(Exception):
            pass
        c = classify_error(APITimeoutError("timed out"))
        assert c.reason == FailoverReason.timeout
        assert c.retryable

    def test_unknown_error(self):
        c = classify_error(ValueError("something weird"))
        assert c.reason == FailoverReason.unknown
        assert c.retryable

    def test_error_code_classification(self):
        c = classify_error(_FakeError(
            "error", body={"error": {"code": "context_length_exceeded"}}
        ))
        assert c.reason == FailoverReason.context_overflow

    def test_billing_pattern_no_status(self):
        c = classify_error(_FakeError("your credits have been exhausted"))
        assert c.reason == FailoverReason.billing
        assert not c.retryable

    def test_auth_pattern_no_status(self):
        c = classify_error(_FakeError("invalid api key provided"))
        assert c.reason == FailoverReason.auth

    def test_disconnect_large_session(self):
        c = classify_error(
            _FakeError("server disconnected"),
            approx_tokens=150_000,
        )
        assert c.reason == FailoverReason.context_overflow
        assert c.should_compress

    def test_disconnect_small_session(self):
        c = classify_error(
            _FakeError("server disconnected"),
            approx_tokens=1000,
        )
        assert c.reason == FailoverReason.timeout

    def test_generic_400_large_session(self):
        c = classify_error(
            _FakeError("error", status_code=400),
            approx_tokens=100_000,
        )
        assert c.reason == FailoverReason.context_overflow

    def test_403_key_limit(self):
        c = classify_error(_FakeError("key limit exceeded", status_code=403))
        assert c.reason == FailoverReason.billing

    def test_is_auth_property(self):
        c = classify_error(_FakeError("unauthorized", status_code=401))
        assert c.is_auth

    def test_provider_model_passthrough(self):
        c = classify_error(
            _FakeError("error", status_code=500),
            provider="anthropic", model="claude-opus-4-6",
        )
        assert c.provider == "anthropic"
        assert c.model == "claude-opus-4-6"

    def test_status_from_cause_chain(self):
        inner = _FakeError("inner", status_code=429)
        outer = Exception("outer")
        outer.__cause__ = inner
        c = classify_error(outer)
        assert c.reason == FailoverReason.rate_limit

    def test_body_extraction(self):
        c = classify_error(_FakeError(
            "error",
            body={"error": {"message": "rate limit exceeded", "code": "rate_limit_exceeded"}},
        ))
        assert c.reason == FailoverReason.rate_limit


# --- Retry ---

class TestJitteredBackoff:
    def test_first_attempt(self):
        delay = jittered_backoff(1, base_delay=5.0)
        assert 5.0 <= delay <= 7.5  # base + up to 50% jitter

    def test_increases(self):
        d1 = jittered_backoff(1, base_delay=5.0, jitter_ratio=0.0)
        d2 = jittered_backoff(2, base_delay=5.0, jitter_ratio=0.0)
        d3 = jittered_backoff(3, base_delay=5.0, jitter_ratio=0.0)
        assert d1 < d2 < d3

    def test_max_cap(self):
        delay = jittered_backoff(100, base_delay=5.0, max_delay=60.0)
        assert delay <= 60.0 * 1.5 + 1  # max + jitter

    def test_decorrelation(self):
        delays = [jittered_backoff(1) for _ in range(10)]
        assert len(set(delays)) > 1  # Not all the same


class TestRetryWithBackoff:
    def test_success_no_retry(self):
        call_count = 0
        async def fn():
            nonlocal call_count
            call_count += 1
            return "ok"

        result = asyncio.run(retry_with_backoff(fn, max_retries=3))
        assert result == "ok"
        assert call_count == 1

    def test_retry_on_transient(self):
        call_count = 0
        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise _FakeError("overloaded", status_code=503)
            return "ok"

        result = asyncio.run(retry_with_backoff(
            fn, max_retries=3, base_delay=0.01,
        ))
        assert result == "ok"
        assert call_count == 3

    def test_no_retry_on_permanent(self):
        async def fn():
            raise _FakeError("model not found", status_code=404)

        with pytest.raises(_FakeError):
            asyncio.run(retry_with_backoff(fn, max_retries=3))

    def test_exhausted_retries(self):
        async def fn():
            raise _FakeError("server error", status_code=500)

        with pytest.raises(_FakeError):
            asyncio.run(retry_with_backoff(
                fn, max_retries=2, base_delay=0.01,
            ))

    def test_on_retry_callback(self):
        retries = []
        async def fn():
            if len(retries) < 2:
                raise _FakeError("overloaded", status_code=503)
            return "ok"

        def on_retry(classified, attempt, delay):
            retries.append((classified.reason, attempt))

        asyncio.run(retry_with_backoff(
            fn, max_retries=3, base_delay=0.01,
            on_retry=on_retry,
        ))
        assert len(retries) == 2
        assert all(r[0] == FailoverReason.overloaded for r in retries)

    def test_compress_callback(self):
        compressed = []
        call_count = 0

        async def fn():
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise _FakeError("context length exceeded", status_code=400)
            return "ok"

        async def on_compress():
            compressed.append(True)

        asyncio.run(retry_with_backoff(
            fn, max_retries=3, base_delay=0.01,
            on_compress=on_compress,
        ))
        assert len(compressed) == 1


# --- Credential Pool ---

class TestCredential:
    def test_available_ok(self):
        c = Credential(provider="anthropic", key="sk-123")
        assert c.is_available

    def test_exhausted_not_available(self):
        c = Credential(provider="anthropic", key="sk-123")
        c.mark_exhausted(429, "rate limited")
        assert not c.is_available
        assert c.status == "exhausted"

    def test_cooldown_recovery(self):
        c = Credential(provider="anthropic", key="sk-123")
        c.mark_exhausted(429)
        c.exhausted_at = time.time() - COOLDOWN_SECONDS - 1
        assert c.is_available
        assert c.status == "ok"

    def test_record_use(self):
        c = Credential(provider="anthropic", key="sk-123")
        c.record_use()
        c.record_use()
        assert c.request_count == 2


class TestCredentialPool:
    def test_add_and_get(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1", label="key1")
        pool.add_key("anthropic", "sk-2", label="key2")
        cred = pool.get("anthropic")
        assert cred is not None
        assert cred.key in ("sk-1", "sk-2")

    def test_round_robin(self):
        pool = CredentialPool(strategy=RotationStrategy.ROUND_ROBIN)
        pool.add_key("anthropic", "sk-1")
        pool.add_key("anthropic", "sk-2")
        keys = [pool.get("anthropic").key for _ in range(4)]
        assert keys == ["sk-1", "sk-2", "sk-1", "sk-2"]

    def test_least_used(self):
        pool = CredentialPool(strategy=RotationStrategy.LEAST_USED)
        pool.add_key("anthropic", "sk-1")
        pool.add_key("anthropic", "sk-2")
        # First call picks sk-1 (both at 0, first in list)
        c1 = pool.get("anthropic")
        assert c1.key == "sk-1"
        # Second call picks sk-2 (sk-1 now at 1 use)
        c2 = pool.get("anthropic")
        assert c2.key == "sk-2"

    def test_exhausted_rotation(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1")
        pool.add_key("anthropic", "sk-2")
        next_cred = pool.mark_exhausted("anthropic", "sk-1", 429, "rate limited")
        assert next_cred is not None
        assert next_cred.key == "sk-2"

    def test_all_exhausted(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1")
        pool.mark_exhausted("anthropic", "sk-1", 429)
        assert pool.get("anthropic") is None

    def test_unknown_provider(self):
        pool = CredentialPool()
        assert pool.get("unknown") is None

    def test_available_count(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1")
        pool.add_key("anthropic", "sk-2")
        assert pool.available_count("anthropic") == 2
        pool.mark_exhausted("anthropic", "sk-1")
        assert pool.available_count("anthropic") == 1

    def test_status_summary(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1")
        pool.add_key("anthropic", "sk-2")
        pool.mark_exhausted("anthropic", "sk-1")
        summary = pool.status_summary()
        assert summary["anthropic"]["ok"] == 1
        assert summary["anthropic"]["exhausted"] == 1
        assert summary["anthropic"]["total"] == 2

    def test_from_config(self):
        config = {
            "providers": {
                "anthropic": {
                    "api_key": "sk-primary",
                    "extra_keys": [
                        "sk-backup1",
                        {"key": "sk-backup2", "label": "backup2", "priority": 2},
                    ],
                },
                "openai": {
                    "api_key": "sk-openai",
                },
            },
        }
        pool = CredentialPool.from_config(config)
        assert pool.total_count("anthropic") == 3
        assert pool.total_count("openai") == 1

    def test_from_config_env_var_skipped(self):
        config = {
            "providers": {
                "anthropic": {"api_key": "${ANTHROPIC_API_KEY}"},
            },
        }
        pool = CredentialPool.from_config(config)
        assert pool.total_count("anthropic") == 0

    def test_providers_list(self):
        pool = CredentialPool()
        pool.add_key("anthropic", "sk-1")
        pool.add_key("openai", "sk-2")
        assert sorted(pool.providers()) == ["anthropic", "openai"]

    def test_mark_ok(self):
        c = Credential(provider="anthropic", key="sk-1")
        c.mark_exhausted(429)
        assert not c.is_available
        c.mark_ok()
        assert c.is_available
