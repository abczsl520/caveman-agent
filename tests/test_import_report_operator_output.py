from pathlib import Path

from caveman.import_.base import ImportItem, ImportManifest, ImportResult
from caveman.import_.report import format_detect_report, format_manifest_report, format_result_report


def test_import_detect_report_escapes_source_labels():
    source = "Hermes\nSPOOF_SOURCE\x1b[31m"

    report = format_detect_report({source: True})

    assert "Hermes\\nSPOOF_SOURCE\\x1b[31m" in report
    assert "\nSPOOF_SOURCE" not in report
    assert "\x1b" not in report


def test_import_manifest_report_escapes_source_path_and_skip_reason():
    manifest = ImportManifest(
        source="OpenClaw\nSPOOF_MANIFEST\x1b[32m",
        items=[
            ImportItem(
                source_path=Path("safe\nSPOOF_PATH\x1b[33m.md"),
                target_type="memory",
                skip_reason="bad\nSPOOF_REASON\x1b[34m",
            )
        ],
    )

    report = format_manifest_report(manifest)

    assert "OpenClaw\\nSPOOF_MANIFEST\\x1b[32m" in report
    assert "safe\\nSPOOF_PATH\\x1b[33m.md" in report
    assert "bad\\nSPOOF_REASON\\x1b[34m" in report
    assert "\nSPOOF_MANIFEST" not in report
    assert "\nSPOOF_PATH" not in report
    assert "\nSPOOF_REASON" not in report
    assert "\x1b" not in report


def test_import_manifest_report_escapes_target_type_labels():
    manifest = ImportManifest(
        source="OpenClaw",
        items=[
            ImportItem(
                source_path=Path("safe.md"),
                target_type="memory\nSPOOF_TARGET\x1b[31m",
            )
        ],
    )

    report = format_manifest_report(manifest)

    assert "memory\\nSPOOF_TARGET\\x1b[31m" in report
    assert "\nSPOOF_TARGET" not in report
    assert "\x1b" not in report


def test_import_result_report_escapes_warnings_and_details():
    result = ImportResult(
        warnings=["warn\nSPOOF_WARNING\x1b[35m"],
        details=["detail\nSPOOF_DETAIL\x1b[36m"],
    )

    report = format_result_report(result)

    assert "warn\\nSPOOF_WARNING\\x1b[35m" in report
    assert "detail\\nSPOOF_DETAIL\\x1b[36m" in report
    assert "\nSPOOF_WARNING" not in report
    assert "\nSPOOF_DETAIL" not in report
    assert "\x1b" not in report
