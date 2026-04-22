"""Service installation — launchd (macOS) and systemd (Linux) support.

Provides `caveman install` and `caveman uninstall` CLI commands to register
the gateway as a system service with automatic restart on crash.

macOS: ~/Library/LaunchAgents/ai.caveman.gateway.plist
Linux: ~/.config/systemd/user/caveman-gateway.service
"""
from __future__ import annotations

import logging
import os
import subprocess
import sys
from pathlib import Path

__all__ = [
    "LAUNCHD_LABEL",
    "SYSTEMD_SERVICE",
    "generate_launchd_plist",
    "install_launchd",
    "uninstall_launchd",
    "generate_systemd_unit",
    "install_systemd",
    "uninstall_systemd",
    "install_service",
    "uninstall_service",
]


logger = logging.getLogger("caveman.gateway.service")

LAUNCHD_LABEL = "ai.caveman.gateway"
SYSTEMD_SERVICE = "caveman-gateway"


# ── launchd (macOS) ──

def _launchd_plist_path() -> Path:
    return Path.home() / "Library" / "LaunchAgents" / f"{LAUNCHD_LABEL}.plist"


def _launchd_domain() -> str:
    return f"gui/{os.getuid()}"


def generate_launchd_plist() -> str:
    """Generate a launchd plist for the Caveman gateway."""
    caveman_bin = _find_caveman_bin()
    home = str(Path.home())
    caveman_home = os.environ.get("CAVEMAN_HOME", f"{home}/.caveman")

    # Build ProgramArguments — handle fallback to python -m
    if caveman_bin.endswith("python") or caveman_bin.endswith("python3"):
        args_xml = f"""        <string>{caveman_bin}</string>
        <string>-m</string>
        <string>caveman.cli.main</string>
        <string>serve</string>"""
    else:
        args_xml = f"""        <string>{caveman_bin}</string>
        <string>serve</string>"""

    # Minimal PATH: only what's needed to find the executable
    bin_dir = str(Path(caveman_bin).parent)
    minimal_path = f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"

    # Log paths: user-specific to avoid multi-user conflicts
    log_dir = f"{caveman_home}/logs"

    return f"""<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN"
  "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
    <key>Label</key>
    <string>{LAUNCHD_LABEL}</string>

    <key>ProgramArguments</key>
    <array>
{args_xml}
    </array>

    <key>EnvironmentVariables</key>
    <dict>
        <key>CAVEMAN_HOME</key>
        <string>{caveman_home}</string>
        <key>PATH</key>
        <string>{minimal_path}</string>
        <key>HOME</key>
        <string>{home}</string>
    </dict>

    <key>WorkingDirectory</key>
    <string>{home}</string>

    <key>RunAtLoad</key>
    <true/>

    <key>KeepAlive</key>
    <dict>
        <key>SuccessfulExit</key>
        <false/>
    </dict>

    <key>ThrottleInterval</key>
    <integer>5</integer>

    <key>StandardOutPath</key>
    <string>{log_dir}/gateway-stdout.log</string>

    <key>StandardErrorPath</key>
    <string>{log_dir}/gateway-stderr.log</string>

    <key>ProcessType</key>
    <string>Background</string>
</dict>
</plist>
"""


def install_launchd() -> dict:
    """Install the Caveman gateway as a launchd service."""
    plist_path = _launchd_plist_path()
    plist_content = generate_launchd_plist()

    # Write plist
    plist_path.parent.mkdir(parents=True, exist_ok=True)
    plist_path.write_text(plist_content, encoding="utf-8")
    logger.info("Wrote plist: %s", plist_path)

    # Bootstrap (register with launchd)
    domain = _launchd_domain()
    try:
        # First try to bootout any existing service
        subprocess.run(
            ["launchctl", "bootout", f"{domain}/{LAUNCHD_LABEL}"],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass  # intentional: OSError suppressed

    try:
        result = subprocess.run(
            ["launchctl", "bootstrap", domain, str(plist_path)],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode != 0:
            return {"ok": False, "detail": result.stderr.strip() or f"exit {result.returncode}"}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return {"ok": False, "detail": str(e)}

    # Start the service
    try:
        subprocess.run(
            ["launchctl", "kickstart", f"{domain}/{LAUNCHD_LABEL}"],
            capture_output=True, timeout=10,
        )
    except (subprocess.TimeoutExpired, OSError):
        pass  # intentional: OSError suppressed

    return {"ok": True, "plist": str(plist_path), "label": LAUNCHD_LABEL}


def uninstall_launchd() -> dict:
    """Uninstall the Caveman gateway launchd service."""
    domain = _launchd_domain()
    target = f"{domain}/{LAUNCHD_LABEL}"

    try:
        result = subprocess.run(
            ["launchctl", "bootout", target],
            capture_output=True, text=True, timeout=30,
        )
        if result.returncode not in (0, 3, 113):  # 3/113 = already unloaded
            return {"ok": False, "detail": result.stderr.strip()}
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError) as e:
        return {"ok": False, "detail": str(e)}

    # Remove plist file
    plist_path = _launchd_plist_path()
    try:
        plist_path.unlink(missing_ok=True)
    except OSError:
        pass  # intentional: OSError suppressed

    return {"ok": True}


# ── systemd (Linux) ──

def _systemd_unit_path() -> Path:
    return Path.home() / ".config" / "systemd" / "user" / f"{SYSTEMD_SERVICE}.service"


def generate_systemd_unit() -> str:
    """Generate a systemd user unit for the Caveman gateway."""
    caveman_bin = _find_caveman_bin()
    from caveman.paths import CAVEMAN_HOME
    caveman_home = str(CAVEMAN_HOME)

    # Handle fallback to python -m
    if caveman_bin.endswith("python") or caveman_bin.endswith("python3"):
        exec_start = f"{caveman_bin} -m caveman.cli.main serve"
    else:
        exec_start = f"{caveman_bin} serve"

    # Minimal PATH
    bin_dir = str(Path(caveman_bin).parent)
    minimal_path = f"{bin_dir}:/usr/local/bin:/usr/bin:/bin"

    return f"""[Unit]
Description=Caveman AI Agent Gateway
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
ExecStart={exec_start}
Environment=CAVEMAN_HOME={caveman_home}
Environment=PATH={minimal_path}
Restart=on-failure
RestartSec=5
RestartForceExitStatus=75
TimeoutStopSec=310

[Install]
WantedBy=default.target
"""


def install_systemd() -> dict:
    """Install the Caveman gateway as a systemd user service."""
    unit_path = _systemd_unit_path()
    unit_content = generate_systemd_unit()

    unit_path.parent.mkdir(parents=True, exist_ok=True)
    unit_path.write_text(unit_content, encoding="utf-8")
    logger.info("Wrote unit: %s", unit_path)

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                        capture_output=True, timeout=10, check=True)
        subprocess.run(["systemctl", "--user", "enable", SYSTEMD_SERVICE],
                        capture_output=True, timeout=10, check=True)
        subprocess.run(["systemctl", "--user", "start", SYSTEMD_SERVICE],
                        capture_output=True, timeout=10, check=True)
    except (subprocess.CalledProcessError, subprocess.TimeoutExpired,
            FileNotFoundError, OSError) as e:
        return {"ok": False, "detail": str(e)}

    return {"ok": True, "unit": str(unit_path), "service": SYSTEMD_SERVICE}


def uninstall_systemd() -> dict:
    """Uninstall the Caveman gateway systemd user service."""
    try:
        subprocess.run(["systemctl", "--user", "stop", SYSTEMD_SERVICE],
                        capture_output=True, timeout=30)
        subprocess.run(["systemctl", "--user", "disable", SYSTEMD_SERVICE],
                        capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # intentional: FileNotFoundError/OSError suppressed

    unit_path = _systemd_unit_path()
    try:
        unit_path.unlink(missing_ok=True)
    except OSError:
        pass  # intentional: OSError suppressed

    try:
        subprocess.run(["systemctl", "--user", "daemon-reload"],
                        capture_output=True, timeout=10)
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        pass  # intentional: FileNotFoundError/OSError suppressed

    return {"ok": True}


# ── Cross-platform ──

def install_service() -> dict:
    """Install the gateway service using the platform's service manager.

    Stops any existing non-managed gateway first to avoid dual-process conflicts.
    """
    from caveman.gateway.status import get_running_pid, terminate_pid, _is_pid_alive

    existing = get_running_pid()
    if existing:
        logger.info("Stopping existing gateway (PID %d) before install", existing)
        try:
            terminate_pid(existing)
            import time
            # Wait up to 5s for graceful shutdown, then force kill
            for _ in range(10):
                time.sleep(0.5)
                if not _is_pid_alive(existing):
                    break
            else:
                logger.warning("Gateway PID %d didn't exit, force killing", existing)
                try:
                    terminate_pid(existing, force=True)
                    time.sleep(1)
                except (ProcessLookupError, PermissionError) as exc:
                    logger.debug("install_service: suppressed %s", exc)
        except (ProcessLookupError, PermissionError) as exc:
            logger.debug("install_service: suppressed %s", exc)

    if sys.platform == "darwin":
        return install_launchd()
    elif sys.platform == "linux":
        return install_systemd()
    return {"ok": False, "detail": f"Unsupported platform: {sys.platform}"}


def uninstall_service() -> dict:
    """Uninstall the gateway service."""
    if sys.platform == "darwin":
        return uninstall_launchd()
    elif sys.platform == "linux":
        return uninstall_systemd()
    return {"ok": False, "detail": f"Unsupported platform: {sys.platform}"}


def _find_caveman_bin() -> str:
    """Find the caveman executable path.

    Returns a single executable path (no spaces). If the caveman script
    isn't found, returns the Python interpreter path — the caller must
    handle the 'serve' argument separately.
    """
    # Check if running from a venv
    venv = os.environ.get("VIRTUAL_ENV")
    if venv:
        candidate = Path(venv) / "bin" / "caveman"
        if candidate.exists():
            return str(candidate)

    # Check sys.executable's directory
    bin_dir = Path(sys.executable).parent
    candidate = bin_dir / "caveman"
    if candidate.exists():
        return str(candidate)

    # Fallback: use which
    import shutil
    found = shutil.which("caveman")
    if found:
        return found

    # Last resort: return Python interpreter (caller adds -m caveman.cli.main serve)
    return sys.executable
