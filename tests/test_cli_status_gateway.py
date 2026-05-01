"""Tests for CLI status dashboard gateway diagnostics."""
from __future__ import annotations

import json


def test_status_text_gateway_scans_only_current_startup_window(tmp_path, monkeypatch):
    from caveman.cli.status import status_text

    pidfile = tmp_path / "gateway.pid"
    logfile = tmp_path / "gateway.log"
    pidfile.write_text(json.dumps({
        "pid": 321,
        "kind": "caveman-gateway",
        "started_at": "2026-04-27T03:45:23+00:00",
    }), encoding="utf-8")
    logfile.write_text(
        "\n".join([
            "2026-04-26 19:30:05 caveman.security.permissions WARNING Permission DENIED old",
            "2026-04-27 11:45:23 caveman.gateway INFO PID file written: ~/.caveman/gateway.pid (PID 321)",
            "2026-04-27 11:45:24 caveman.gateway INFO Discord connected: wildman#1416",
            "2026-04-27 11:45:25 caveman.gateway INFO Synced 59 slash commands to guild wildman的服务器",
        ]),
        encoding="utf-8",
    )
    monkeypatch.setattr("caveman.gateway.log_diagnostics._default_pidfile", lambda: pidfile)
    monkeypatch.setattr("caveman.gateway.log_diagnostics._default_logfile", lambda: logfile)
    monkeypatch.setattr("caveman.gateway.status.get_running_pid", lambda: 321)

    text = status_text(include_gateway=True)

    assert "Gateway: running (PID 321)" in text
    assert "Gateway log window: bounded via 'pid_marker'" in text
    assert "Gateway log alerts: none" in text
    assert "Discord connected ✅" in text
    assert "Slash commands synced ✅" in text
    assert "Permission DENIED old" not in text


def test_status_text_gateway_escapes_log_diagnostic_labels(monkeypatch):
    """Log-derived gateway diagnostics are operator-facing and must not spoof lines."""
    from caveman.cli import status

    monkeypatch.setattr("caveman.gateway.status.get_running_pid", lambda: 321)
    monkeypatch.setattr(
        "caveman.gateway.log_diagnostics.scan_current_startup_log",
        lambda *, expected_pid: {
            "bounded": True,
            "boundary": "pid_marker\nSPOOF_BOUNDARY\x1b[31m",
            "line_count": 4,
            "patterns": {"ERROR\nSPOOF_PATTERN\x1b[31m": {"count": 2, "samples": []}},
            "healthy_markers": {
                "discord_connected": True,
                "slash_commands_synced": True,
            },
        },
    )

    text = status.status_text(include_gateway=True)

    assert "'pid_marker\\nSPOOF_BOUNDARY\\x1b[31m'" in text
    assert "'ERROR\\nSPOOF_PATTERN\\x1b[31m'=2" in text
    assert "\nSPOOF_BOUNDARY" not in text
    assert "\nSPOOF_PATTERN" not in text
    assert "\x1b[31m" not in text
