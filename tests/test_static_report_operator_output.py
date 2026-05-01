"""Operator-output boundary tests for static audit reports."""

from caveman.cli.audit import run_audit
from caveman.cli.code_health import format_report


def test_code_health_report_escapes_category_and_issue_labels():
    result = {
        "file_size\nSPOOF_CATEGORY\x1b[31m": [
            "caveman/evil.py: 999 lines\nSPOOF_ISSUE\x1b[32m"
        ],
    }

    report = format_report(result)

    assert "'file_size\\nSPOOF_CATEGORY\\x1b[31m': ❌ 1" in report
    assert "'caveman/evil.py: 999 lines\\nSPOOF_ISSUE\\x1b[32m'" in report
    assert "\nSPOOF_CATEGORY" not in report
    assert "\nSPOOF_ISSUE" not in report


def test_audit_report_escapes_check_and_issue_labels(monkeypatch):
    monkeypatch.setattr("caveman.cli.audit._find_python_files", lambda: [])
    monkeypatch.setattr("caveman.cli.audit.check_encoding", lambda files: ["open.py\nSPOOF_OPEN\x1b[33m"])
    monkeypatch.setattr("caveman.cli.audit.check_uuid_truncation", lambda files: [])
    monkeypatch.setattr("caveman.cli.audit.check_swallowed_exceptions", lambda files: [])
    monkeypatch.setattr("caveman.cli.audit.check_file_size", lambda files: [])

    report = run_audit()

    assert "'encoding': ❌ 1" in report
    assert "'open.py\\nSPOOF_OPEN\\x1b[33m'" in report
    assert "\nSPOOF_OPEN" not in report
