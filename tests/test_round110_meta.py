"""Tests for Round 110 — meta-flywheel: success detection + incremental lint."""
import pytest
from caveman.utils import detect_success as _detect_success


# ── Success detection tests ──

class TestDetectSuccess:
    """The old `"error" not in text[:100]` was catastrophically wrong.
    These tests ensure the new multi-signal detection works correctly.
    """

    def test_done_claim_alone_is_not_success(self):
        assert _detect_success("Done! I've created the file successfully.") is False

    def test_objective_tests_passed_is_success(self):
        assert _detect_success("Verified with pytest: 12 passed in 0.31s") is True

    def test_obvious_failure(self):
        assert _detect_success("ERROR: could not connect to database") is False

    def test_fixed_error_claim_without_verification_is_not_success(self):
        """'I fixed the error' is a claim, not objective completion evidence."""
        assert _detect_success("I fixed the error in line 42.") is False

    def test_fixed_error_with_tests_is_success(self):
        assert _detect_success("I fixed the error in line 42. All tests pass now.") is True

    def test_resolved_error_claim_without_verification_is_not_success(self):
        assert _detect_success("I resolved the error by updating the config.") is False

    def test_found_error_is_not_success_without_verification(self):
        assert _detect_success("I found the error — it was a missing import. Here's the fix.") is False

    def test_empty_is_failure(self):
        assert _detect_success("") is False

    def test_normal_output_is_not_success_without_verification(self):
        """Agent producing normal output is not verified task completion."""
        assert _detect_success("Here is the implementation you requested.") is False

    def test_here_are_results_with_tests_passed(self):
        assert _detect_success("Here are the test results: all 15 tests pass.") is True

    def test_unable_to_is_failure(self):
        assert _detect_success("I was unable to complete the task due to permissions.") is False

    def test_sorry_cant_is_failure(self):
        assert _detect_success("Sorry, I can't access that file.") is False

    def test_mixed_signals_failure_wins_without_objective_evidence(self):
        text = "I fixed the TypeError and created the new module."
        assert _detect_success(text) is False

    def test_mixed_signals_with_tests_passed_is_success(self):
        text = "I fixed the TypeError and created the new module. All tests pass."
        assert _detect_success(text) is True

    def test_no_signals_defaults_to_failure(self):
        """Neutral output (no strong evidence) must not inflate success."""
        assert _detect_success("The function returns a list of integers.") is False

    def test_traceback_is_failure(self):
        assert _detect_success("Traceback (most recent call last):\n  File...") is False

    def test_completed_with_error_mention_needs_objective_evidence(self):
        """Discussing errors plus a done token is still only a claim."""
        text = "I debugged the error and found it was caused by a race condition. Fixed now. ✅"
        assert _detect_success(text) is False

    def test_completed_with_error_and_tests_is_success(self):
        text = "I debugged the error and found it was caused by a race condition. 8 passed."
        assert _detect_success(text) is True


# ── Incremental lint tests ──

@pytest.mark.asyncio
async def test_lint_incremental_scan(tmp_path):
    """Lint should scan incrementally by default, full every N scans."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType
    from caveman.engines.lint import LintEngine

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "test.db")

    # Store some initial memories
    for i in range(5):
        await mm.store(f"Memory number {i}", MemoryType.SEMANTIC)

    lint = LintEngine(memory_manager=mm)

    # First scan: full (because _last_scan_count == 0)
    report1 = await lint.scan()
    assert report1.scanned == 5
    assert lint._last_scan_count == 5

    # Add 2 more memories
    await mm.store("New memory 1", MemoryType.SEMANTIC)
    await mm.store("New memory 2", MemoryType.SEMANTIC)

    # Second scan: incremental (only new memories)
    report2 = await lint.scan()
    assert report2.scanned == 2  # Only the 2 new ones
    assert lint._last_scan_count == 7


@pytest.mark.asyncio
async def test_lint_full_scan_periodic(tmp_path):
    """Lint should do full scan every N incremental scans."""
    from caveman.memory.manager import MemoryManager
    from caveman.memory.types import MemoryType
    from caveman.engines.lint import LintEngine

    mm = MemoryManager.with_sqlite(base_dir=tmp_path, db_path=tmp_path / "test.db")
    for i in range(3):
        await mm.store(f"Memory {i}", MemoryType.SEMANTIC)

    lint = LintEngine(memory_manager=mm)
    lint._full_scan_interval = 3  # Full scan every 3rd scan

    # Scan 1: full (first scan)
    r1 = await lint.scan()
    assert r1.scanned == 3

    # Scan 2: incremental (no new memories → 0)
    r2 = await lint.scan()
    assert r2.scanned == 0

    # Scan 3: full (every 3rd scan)
    r3 = await lint.scan()
    assert r3.scanned == 3  # Full scan


# ── detect_outcome tests (Round 128) ──

def test_detect_outcome_success():
    from caveman.utils import detect_outcome
    assert detect_outcome("Verified: all tests pass. 15 passed.") == "success"


def test_detect_outcome_done_claim_is_partial():
    from caveman.utils import detect_outcome
    assert detect_outcome("Done! I created the file. ✅") == "partial"


def test_detect_outcome_failure():
    from caveman.utils import detect_outcome
    assert detect_outcome("ERROR: could not connect") == "failure"


def test_detect_outcome_partial():
    from caveman.utils import detect_outcome
    # Has both success and failure signals, but failure wins
    # yet there ARE success signals → partial
    from caveman.utils import detect_outcome
    assert detect_outcome("I completed part of the task but encountered an ERROR") in ("partial", "success")


def test_detect_outcome_empty():
    from caveman.utils import detect_outcome
    assert detect_outcome("") == "failure"
