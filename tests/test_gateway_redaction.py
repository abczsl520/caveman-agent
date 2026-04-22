"""Tests for gateway redaction engine."""
import pytest

from caveman.gateway.redaction import (
    redact_secrets,
    redact_pii,
    redact_all,
    detect_secrets,
    detect_pii,
    is_safe_for_external,
)


class TestRedactSecrets:
    def test_openai_key(self):
        text = "key is sk-abcdefghijklmnopqrstuvwxyz1234"
        assert "[REDACTED]" in redact_secrets(text)
        assert "sk-abc" not in redact_secrets(text)

    def test_anthropic_key(self):
        text = "sk-ant-abcdefghijklmnopqrstuvwxyz1234"
        assert "[REDACTED]" in redact_secrets(text)

    def test_github_token(self):
        text = "ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij"
        assert "[REDACTED]" in redact_secrets(text)

    def test_aws_access_key(self):
        text = "AKIAIOSFODNN7EXAMPLE"
        assert "[REDACTED]" in redact_secrets(text)

    def test_bearer_token(self):
        text = "Authorization: Bearer eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.test"
        result = redact_secrets(text)
        assert "eyJhbG" not in result

    def test_private_key(self):
        text = "-----BEGIN PRIVATE KEY-----\nMIIEvgIBADANBg..."
        assert "[REDACTED]" in redact_secrets(text)

    def test_connection_string(self):
        text = "mongodb://user:pass@host:27017/db"
        assert "[REDACTED]" in redact_secrets(text)

    def test_clean_text_unchanged(self):
        text = "This is a normal message with no secrets."
        assert redact_secrets(text) == text

    def test_custom_replacement(self):
        text = "sk-abcdefghijklmnopqrstuvwxyz1234"
        assert "***" in redact_secrets(text, replacement="***")


class TestRedactPII:
    def test_email(self):
        text = "Contact me at user@example.com"
        assert "[PII]" in redact_pii(text)
        assert "user@example.com" not in redact_pii(text)

    def test_us_phone(self):
        text = "Call 555-123-4567"
        assert "[PII]" in redact_pii(text)

    def test_intl_phone(self):
        text = "Call +86-13800138000"
        assert "[PII]" in redact_pii(text)

    def test_ssn(self):
        text = "SSN: 123-45-6789"
        assert "[PII]" in redact_pii(text)

    def test_credit_card(self):
        text = "Card: 4111-1111-1111-1111"
        assert "[PII]" in redact_pii(text)

    def test_ip_address(self):
        text = "Server at 192.168.1.100"
        assert "[PII]" in redact_pii(text)


class TestRedactAll:
    def test_mixed_content(self):
        text = "Key: sk-abcdefghijklmnopqrstuvwxyz1234, email: user@example.com"
        result = redact_all(text)
        assert "sk-abc" not in result
        assert "user@example.com" not in result


class TestDetection:
    def test_detect_secrets_returns_findings(self):
        text = "sk-abcdefghijklmnopqrstuvwxyz1234"
        findings = detect_secrets(text)
        assert len(findings) >= 1
        assert findings[0]["type"] == "api_key_openai"

    def test_detect_pii_returns_findings(self):
        text = "user@example.com"
        findings = detect_pii(text)
        assert len(findings) >= 1
        assert findings[0]["type"] == "email"

    def test_is_safe_clean_text(self):
        assert is_safe_for_external("Hello, how are you?")

    def test_is_safe_with_secret(self):
        assert not is_safe_for_external("sk-abcdefghijklmnopqrstuvwxyz1234")

    def test_is_safe_with_pii(self):
        assert not is_safe_for_external("user@example.com")
