"""Tests for website access policy."""
import pytest

from caveman.security.website_policy import check_url, _matches_any, clear_cache


@pytest.fixture(autouse=True)
def _clear():
    clear_cache()
    yield
    clear_cache()


class TestCheckUrl:
    def test_disabled_allows_all(self):
        policy = {"enabled": False, "blocked_domains": ["evil.com"]}
        allowed, reason = check_url("https://evil.com", config=policy)
        assert allowed is True

    def test_blocklist_blocks(self):
        policy = {"enabled": True, "blocked_domains": ["evil.com"], "allowed_domains": [], "shared_files": []}
        allowed, reason = check_url("https://evil.com/page", config=policy)
        assert allowed is False
        assert "blocked" in reason

    def test_blocklist_allows_others(self):
        policy = {"enabled": True, "blocked_domains": ["evil.com"], "allowed_domains": [], "shared_files": []}
        allowed, _ = check_url("https://good.com", config=policy)
        assert allowed is True

    def test_blocklist_subdomain(self):
        policy = {"enabled": True, "blocked_domains": ["evil.com"], "allowed_domains": [], "shared_files": []}
        allowed, _ = check_url("https://sub.evil.com/page", config=policy)
        assert allowed is False

    def test_allowlist_mode(self):
        policy = {"enabled": True, "blocked_domains": [], "allowed_domains": ["trusted.com"], "shared_files": []}
        allowed, _ = check_url("https://trusted.com", config=policy)
        assert allowed is True
        allowed, reason = check_url("https://other.com", config=policy)
        assert allowed is False
        assert "allowlist" in reason

    def test_wildcard_pattern(self):
        policy = {"enabled": True, "blocked_domains": ["*.evil.com"], "allowed_domains": [], "shared_files": []}
        allowed, _ = check_url("https://sub.evil.com", config=policy)
        assert allowed is False
        allowed, _ = check_url("https://evil.com", config=policy)
        assert allowed is True  # *.evil.com doesn't match evil.com itself

    def test_invalid_url(self):
        policy = {"enabled": True, "blocked_domains": [], "allowed_domains": [], "shared_files": []}
        allowed, reason = check_url("not-a-url", config=policy)
        assert allowed is False

    def test_shared_file(self, tmp_path):
        blocklist = tmp_path / "blocklist.txt"
        blocklist.write_text("spam.com\n# comment\nphishing.org\n")
        policy = {
            "enabled": True,
            "blocked_domains": [],
            "allowed_domains": [],
            "shared_files": [str(blocklist)],
        }
        allowed, _ = check_url("https://spam.com", config=policy)
        assert allowed is False
        allowed, _ = check_url("https://phishing.org", config=policy)
        assert allowed is False
        allowed, _ = check_url("https://safe.com", config=policy)
        assert allowed is True


class TestMatchesAny:
    def test_exact(self):
        assert _matches_any("evil.com", ["evil.com"]) is True

    def test_subdomain(self):
        assert _matches_any("sub.evil.com", ["evil.com"]) is True

    def test_no_match(self):
        assert _matches_any("good.com", ["evil.com"]) is False

    def test_wildcard(self):
        assert _matches_any("sub.evil.com", ["*.evil.com"]) is True

    def test_case_insensitive(self):
        assert _matches_any("evil.com", ["EVIL.COM"]) is True

    def test_empty_patterns(self):
        assert _matches_any("anything.com", []) is False
        assert _matches_any("anything.com", ["", "  "]) is False
