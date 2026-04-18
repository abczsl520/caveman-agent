"""Tests for memory quality gate — the highest-leverage flywheel intervention."""
import pytest
from caveman.memory.quality_gate import check_quality, truncate_if_needed, reset_cache


@pytest.fixture(autouse=True)
def _bypass_quality_gate():
    """Override conftest's bypass — we need the real quality gate here."""
    reset_cache()
    yield
    reset_cache()


class TestQualityGate:
    def test_rejects_too_short(self):
        assert check_quality("hi") is not None
        assert "too_short" in check_quality("hi")

    def test_accepts_normal_content(self):
        assert check_quality("The project uses SQLite + FTS5 for memory storage") is None

    def test_rejects_completed_task_garbage(self):
        assert check_quality("Completed task: test") is not None

    def test_rejects_all_tools_executed(self):
        assert check_quality("All tools executed successfully") is not None

    def test_rejects_generic_qa(self):
        assert check_quality("Task: What is the SOLID principle in software engineering?") is not None
        assert check_quality("Task: Explain the difference between TCP and UDP") is not None
        assert check_quality("Task: Write a Python function that checks palindromes") is not None

    def test_rejects_session_metadata(self):
        assert check_quality("# Session: 2026-04-18 12:00:00 UTC") is not None

    def test_rejects_trivial_results(self):
        assert check_quality("Done!") is not None
        assert check_quality("OK") is not None
        assert check_quality("Success.") is not None

    def test_accepts_project_knowledge(self):
        assert check_quality("The root cause was a missing await in the event handler") is None
        assert check_quality("User preference: always use Chinese for communication") is None

    def test_trusted_bypasses_all_checks(self):
        assert check_quality("hi", trusted=True) is None
        assert check_quality("Completed task: test", trusted=True) is None

    def test_near_duplicate_detection(self):
        content = "The project uses SQLite for persistent storage"
        assert check_quality(content) is None  # First time: OK
        assert check_quality(content) is not None  # Second time: duplicate

    def test_near_duplicate_different_content(self):
        assert check_quality("SQLite is used for memory storage") is None
        assert check_quality("The agent loop orchestrates all phases") is None

    def test_accepts_chinese_content(self):
        assert check_quality("用户偏好中文交流，会用中文提问和沟通") is None

    def test_accepts_error_context(self):
        assert check_quality("When trying to read config.yaml, got FileNotFoundError") is None


class TestTruncate:
    def test_short_content_unchanged(self):
        assert truncate_if_needed("short") == "short"

    def test_long_content_truncated(self):
        long = "x" * 5000
        result = truncate_if_needed(long)
        assert len(result) <= 3003  # 3000 + "..."

    def test_truncates_at_sentence_boundary(self):
        content = "First sentence. " * 200  # ~3200 chars
        result = truncate_if_needed(content)
        assert result.endswith(".")
        assert len(result) <= 3000
