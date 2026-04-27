"""Tests for gateway current-startup log diagnostics."""
from __future__ import annotations

import json
from pathlib import Path


def test_gateway_log_scan_ignores_pre_startup_historical_failures(tmp_path):
    from caveman.gateway.log_diagnostics import scan_current_startup_log

    pidfile = tmp_path / "gateway.pid"
    logfile = tmp_path / "gateway.log"
    pidfile.write_text(json.dumps({
        "pid": 123,
        "kind": "caveman-gateway",
        "started_at": "2026-04-27T03:45:23.885094+00:00",
    }), encoding="utf-8")
    logfile.write_text(
        "\n".join([
            "2026-04-26 19:30:05 caveman.security.permissions WARNING Permission DENIED for 'bash_write'",
            "2026-04-26 19:31:51 caveman.agent.bg_tasks WARNING Lint scan failed: no such column: last_accessed",
            "2026-04-27 11:45:23 caveman.gateway INFO PID file written: ~/.caveman/gateway.pid (PID 123)",
            "2026-04-27 11:45:24 caveman.gateway INFO Discord connected: wildman#1416",
            "2026-04-27 11:45:25 caveman.gateway INFO Synced 59 slash commands to guild wildman的服务器",
        ]),
        encoding="utf-8",
    )

    report = scan_current_startup_log(pidfile=pidfile, logfile=logfile)

    assert report["startup_line_index"] == 2
    assert report["line_count"] == 3
    assert report["patterns"]["Permission DENIED"]["count"] == 0
    assert report["patterns"]["no such column"]["count"] == 0
    assert report["patterns"]["Traceback"]["count"] == 0
    assert report["healthy_markers"]["discord_connected"] is True
    assert report["healthy_markers"]["slash_commands_synced"] is True


def test_gateway_log_scan_falls_back_to_started_at_when_pid_line_missing(tmp_path):
    from caveman.gateway.log_diagnostics import scan_current_startup_log

    pidfile = tmp_path / "gateway.pid"
    logfile = tmp_path / "gateway.log"
    pidfile.write_text(json.dumps({
        "pid": 456,
        "kind": "caveman-gateway",
        "started_at": "2026-04-27T03:45:23+00:00",
    }), encoding="utf-8")
    logfile.write_text(
        "\n".join([
            "2026-04-27 11:45:22 caveman.gateway ERROR old error before this process",
            "2026-04-27 11:45:23 caveman.gateway INFO Gateway starting (attempt 1/10)",
            "2026-04-27 11:45:24 caveman.gateway ERROR current process error",
        ]),
        encoding="utf-8",
    )

    report = scan_current_startup_log(pidfile=pidfile, logfile=logfile)

    assert report["startup_line_index"] == 1
    assert report["line_count"] == 2
    assert report["patterns"]["ERROR"]["count"] == 1
    assert "current process error" in report["patterns"]["ERROR"]["samples"][0]


def test_gateway_log_scan_marks_missing_pidfile_unbounded(tmp_path):
    from caveman.gateway.log_diagnostics import scan_current_startup_log

    logfile = tmp_path / "gateway.log"
    logfile.write_text(
        "\n".join([
            "2026-04-27 11:45:24 caveman.gateway ERROR unknown",
            "2026-04-27 11:45:25 caveman.gateway INFO Discord connected: wildman#1416",
            "2026-04-27 11:45:26 caveman.gateway INFO Synced 59 slash commands to guild wildman的服务器",
        ]),
        encoding="utf-8",
    )

    report = scan_current_startup_log(pidfile=tmp_path / "missing.pid", logfile=logfile)

    assert report["bounded"] is False
    assert report["startup_line_index"] == 0
    assert report["patterns"]["ERROR"]["count"] == 1
    assert report["healthy_markers"]["discord_connected"] is False
    assert report["healthy_markers"]["slash_commands_synced"] is False


def test_gateway_log_scan_uses_latest_pid_marker_for_reused_pid(tmp_path):
    from caveman.gateway.log_diagnostics import scan_current_startup_log

    pidfile = tmp_path / "gateway.pid"
    logfile = tmp_path / "gateway.log"
    pidfile.write_text(json.dumps({
        "pid": 777,
        "kind": "caveman-gateway",
        "started_at": "2026-04-27T04:00:00+00:00",
    }), encoding="utf-8")
    logfile.write_text(
        "\n".join([
            "2026-04-26 10:00:00 caveman.gateway INFO PID file written: ~/.caveman/gateway.pid (PID 777)",
            "2026-04-26 10:00:01 caveman.gateway ERROR stale reused-pid error",
            "2026-04-27 12:00:00 caveman.gateway INFO PID file written: ~/.caveman/gateway.pid (PID 777)",
            "2026-04-27 12:00:01 caveman.gateway INFO Discord connected: wildman#1416",
        ]),
        encoding="utf-8",
    )

    report = scan_current_startup_log(pidfile=pidfile, logfile=logfile)

    assert report["boundary"] == "pid_marker"
    assert report["startup_line_index"] == 2
    assert report["patterns"]["ERROR"]["count"] == 0


def test_gateway_log_scan_refuses_mismatched_expected_pid(tmp_path):
    from caveman.gateway.log_diagnostics import scan_current_startup_log

    pidfile = tmp_path / "gateway.pid"
    logfile = tmp_path / "gateway.log"
    pidfile.write_text(json.dumps({"pid": 888, "started_at": "2026-04-27T04:00:00+00:00"}), encoding="utf-8")
    logfile.write_text(
        "\n".join([
            "2026-04-27 12:00:00 caveman.gateway INFO PID file written: ~/.caveman/gateway.pid (PID 888)",
            "2026-04-27 12:00:01 caveman.gateway INFO Discord connected: wildman#1416",
        ]),
        encoding="utf-8",
    )

    report = scan_current_startup_log(pidfile=pidfile, logfile=logfile, expected_pid=999)

    assert report["bounded"] is False
    assert report["boundary"] == "pid_mismatch"
    assert report["healthy_markers"]["discord_connected"] is False
