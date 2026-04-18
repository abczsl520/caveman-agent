"""Tests for P1 — Skill utils + gateway verification."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pytest


class TestFrontmatter:
    def test_parse_basic(self):
        from caveman.skills.utils import parse_frontmatter
        content = "---\nname: test\ndescription: A test skill\n---\n\n# Body"
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "test"
        assert fm["description"] == "A test skill"
        assert "# Body" in body

    def test_parse_no_frontmatter(self):
        from caveman.skills.utils import parse_frontmatter
        fm, body = parse_frontmatter("# Just a heading\nSome text")
        assert fm == {}
        assert "Just a heading" in body

    def test_parse_empty(self):
        from caveman.skills.utils import parse_frontmatter
        fm, body = parse_frontmatter("")
        assert fm == {}

    def test_parse_list_values(self):
        from caveman.skills.utils import parse_frontmatter
        content = "---\nname: test\nplatforms:\n  - macos\n  - linux\n---\n\nbody"
        fm, body = parse_frontmatter(content)
        assert fm["name"] == "test"
        assert "macos" in fm.get("platforms", [])


class TestPlatformMatching:
    def test_no_platforms_matches_all(self):
        from caveman.skills.utils import skill_matches_platform
        assert skill_matches_platform({}) is True
        assert skill_matches_platform({"platforms": []}) is True

    def test_current_platform_matches(self):
        from caveman.skills.utils import skill_matches_platform
        if sys.platform.startswith("darwin"):
            assert skill_matches_platform({"platforms": ["macos"]}) is True
            assert skill_matches_platform({"platforms": ["windows"]}) is False
        elif sys.platform.startswith("linux"):
            assert skill_matches_platform({"platforms": ["linux"]}) is True
            assert skill_matches_platform({"platforms": ["macos"]}) is False

    def test_multiple_platforms(self):
        from caveman.skills.utils import skill_matches_platform
        fm = {"platforms": ["macos", "linux"]}
        # Should match on either macOS or Linux
        if sys.platform.startswith(("darwin", "linux")):
            assert skill_matches_platform(fm) is True


class TestDescription:
    def test_extract_short(self):
        from caveman.skills.utils import extract_skill_description
        assert extract_skill_description({"description": "Short"}) == "Short"

    def test_extract_long_truncates(self):
        from caveman.skills.utils import extract_skill_description
        long_desc = "A" * 100
        result = extract_skill_description({"description": long_desc})
        assert len(result) <= 60
        assert result.endswith("...")

    def test_extract_empty(self):
        from caveman.skills.utils import extract_skill_description
        assert extract_skill_description({}) == ""


class TestConditions:
    def test_extract_conditions(self):
        from caveman.skills.utils import extract_conditions
        fm = {
            "metadata": {
                "caveman": {
                    "requires_tools": ["bash"],
                    "fallback_for_tools": ["web_search"],
                }
            }
        }
        conds = extract_conditions(fm)
        assert "bash" in conds["requires_tools"]
        assert "web_search" in conds["fallback_for_tools"]

    def test_no_metadata(self):
        from caveman.skills.utils import extract_conditions
        conds = extract_conditions({})
        assert conds["requires_tools"] == []


class TestDiscovery:
    def test_iter_skill_files(self, tmp_path):
        from caveman.skills.utils import iter_skill_files
        # Create skill structure
        (tmp_path / "skill_a").mkdir()
        (tmp_path / "skill_a" / "SKILL.md").write_text("---\nname: a\n---\n")
        (tmp_path / "skill_b").mkdir()
        (tmp_path / "skill_b" / "SKILL.md").write_text("---\nname: b\n---\n")
        (tmp_path / ".git").mkdir()
        (tmp_path / ".git" / "SKILL.md").write_text("should be excluded")

        files = iter_skill_files(tmp_path)
        assert len(files) == 2
        names = [f.parent.name for f in files]
        assert "skill_a" in names
        assert ".git" not in names

    def test_discover_skills(self, tmp_path):
        from caveman.skills.utils import discover_skills
        (tmp_path / "my_skill").mkdir()
        (tmp_path / "my_skill" / "SKILL.md").write_text(
            "---\nname: my_skill\ndescription: Test skill\n---\n\n# My Skill"
        )

        skills = discover_skills(skills_dirs=[tmp_path])
        assert len(skills) == 1
        assert skills[0]["name"] == "my_skill"
        assert skills[0]["description"] == "Test skill"

    def test_discover_respects_disabled(self, tmp_path):
        from caveman.skills.utils import discover_skills
        (tmp_path / "s1").mkdir()
        (tmp_path / "s1" / "SKILL.md").write_text("---\nname: s1\n---\n")
        (tmp_path / "s2").mkdir()
        (tmp_path / "s2" / "SKILL.md").write_text("---\nname: s2\n---\n")

        skills = discover_skills(skills_dirs=[tmp_path], disabled={"s1"})
        assert len(skills) == 1
        assert skills[0]["name"] == "s2"

    def test_discover_empty_dir(self, tmp_path):
        from caveman.skills.utils import discover_skills
        skills = discover_skills(skills_dirs=[tmp_path])
        assert skills == []


class TestGatewayImports:
    """Verify gateway modules import cleanly."""

    def test_discord_gateway_import(self):
        from caveman.gateway.discord_gw import DiscordGateway
        gw = DiscordGateway(token="fake")
        assert gw.token == "fake"
        assert gw.prefix == "!cave"

    def test_telegram_gateway_import(self):
        from caveman.gateway.telegram_gw import TelegramGateway
        gw = TelegramGateway(token="fake")
        assert gw.token == "fake"

    def test_runner_import(self):
        from caveman.gateway.runner import run_gateway
        assert callable(run_gateway)

    def test_base_gateway(self):
        from caveman.gateway.base import Gateway
        # Abstract, can't instantiate
        with pytest.raises(TypeError):
            Gateway()


class TestRateLimitTracker:
    def test_parse_headers(self):
        from caveman.providers.rate_limit import parse_rate_limit_headers
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "95",
            "x-ratelimit-reset-requests": "60",
            "x-ratelimit-limit-tokens": "1000000",
            "x-ratelimit-remaining-tokens": "999000",
            "x-ratelimit-reset-tokens": "60",
        }
        state = parse_rate_limit_headers(headers, provider="anthropic")
        assert state is not None
        assert state.requests_min.limit == 100
        assert state.requests_min.remaining == 95
        assert state.tokens_min.limit == 1000000

    def test_no_headers(self):
        from caveman.providers.rate_limit import parse_rate_limit_headers
        assert parse_rate_limit_headers({}) is None

    def test_format(self):
        from caveman.providers.rate_limit import (
            parse_rate_limit_headers, format_rate_limits, format_compact,
        )
        headers = {
            "x-ratelimit-limit-requests": "100",
            "x-ratelimit-remaining-requests": "50",
            "x-ratelimit-reset-requests": "30",
        }
        state = parse_rate_limit_headers(headers)
        text = format_rate_limits(state)
        assert "50.0%" in text
        compact = format_compact(state)
        assert "RPM" in compact

    def test_fmt_count(self):
        from caveman.providers.rate_limit import _fmt_count
        assert _fmt_count(500) == "500"
        assert _fmt_count(1500) == "1.5K"
        assert _fmt_count(7999856) == "8.0M"
