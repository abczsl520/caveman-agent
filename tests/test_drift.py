"""Tests for drift detection — contradiction vs supersession."""
import pytest
from caveman.memory.drift import _is_contradiction, _is_supersede


class TestIsContradiction:
    """True contradictions: simultaneous conflicting claims."""

    def test_likes_dislikes(self):
        assert _is_contradiction("user likes python", "user dislikes python")

    def test_prefers_avoids(self):
        assert _is_contradiction("user prefers vim", "user avoids vim")

    def test_enabled_disabled(self):
        assert _is_contradiction("shield is enabled", "shield is disabled")

    def test_correct_incorrect(self):
        assert _is_contradiction("the answer is correct", "the answer is incorrect")

    # --- False positives that MUST NOT fire ---

    def test_no_fp_use_because(self):
        """'use' inside 'because' must not match."""
        assert not _is_contradiction(
            "we chose this because it's fast",
            "don't use the old approach",
        )

    def test_no_fp_uuid_update(self):
        """Knowledge update, not contradiction."""
        assert not _is_contradiction(
            "uuid should use full uuid4",
            "uuid uses [:8] truncation",
        )

    def test_no_fp_stream_update(self):
        """stream=True vs stream=False is a value change, not opinion conflict."""
        assert not _is_contradiction(
            "stream=true is more reliable",
            "stream=false was used",
        )

    def test_no_fp_wired_vs_dead(self):
        """'wired in' vs 'dead code' is temporal, not contradictory."""
        assert not _is_contradiction(
            "hybridscorer is now wired in",
            "hybridscorer is dead code",
        )

    def test_no_fp_port_change(self):
        """Different port numbers are updates."""
        assert not _is_contradiction(
            "server runs on port 3000",
            "server runs on port 8080",
        )

    def test_no_fp_unrelated(self):
        assert not _is_contradiction("the sky is blue", "python is great")


class TestIsSupersede:
    """Temporal updates: new info replaces old."""

    def test_now_signal(self):
        assert _is_supersede("port is now 3000", "port was 8080")

    def test_fixed_signal(self):
        assert _is_supersede("fixed the uuid collision bug", "uuid has collision risk")

    def test_should_use_signal(self):
        assert _is_supersede("should use full uuid4", "uses truncated uuid")

    def test_refactored_signal(self):
        assert _is_supersede("refactored to stdio transport", "uses http post")

    def test_dead_code_in_old(self):
        assert _is_supersede("hybridscorer is wired in", "hybridscorer is dead code")

    def test_was_in_old(self):
        assert _is_supersede("using stream=true", "stream=false was the default")

    def test_numeric_change(self):
        """Same context, different numbers → supersession."""
        assert _is_supersede("server port 3000", "server port 8080")

    def test_chinese_signals(self):
        assert _is_supersede("现在用 stdio 传输", "之前用 http")

    def test_no_supersede_unrelated(self):
        assert not _is_supersede("the sky is blue", "python is great")

    def test_no_supersede_same_content(self):
        assert not _is_supersede("server runs on port 3000", "server runs on port 3000")
