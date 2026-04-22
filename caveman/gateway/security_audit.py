"""Security Audit — system security scanning and policy enforcement.

Provides security auditing for the agent system including
file permissions, network exposure, and configuration safety.
Core patterns from OpenClaw src/security/ (15K LOC — extracted essentials).
"""
from __future__ import annotations

import logging
import os
import re
import subprocess
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional

__all__ = [
    "AuditFinding",
    "AuditReport",
    "run_audit",
]


logger = logging.getLogger("caveman.gateway.security_audit")


@dataclass
class AuditFinding:
    """A security audit finding."""
    category: str
    severity: str  # info | low | medium | high | critical
    title: str
    description: str = ""
    remediation: str = ""
    path: str = ""


@dataclass
class AuditReport:
    """Complete audit report."""
    findings: List[AuditFinding] = field(default_factory=list)
    scanned_at: float = 0
    duration_ms: float = 0
    score: int = 100  # 0-100, deducted per finding

    @property
    def critical_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "critical")

    @property
    def high_count(self) -> int:
        return sum(1 for f in self.findings if f.severity == "high")

    def compute_score(self) -> int:
        deductions = {"critical": 25, "high": 15, "medium": 5, "low": 2, "info": 0}
        score = 100
        for f in self.findings:
            score -= deductions.get(f.severity, 0)
        self.score = max(0, score)
        return self.score


def run_audit(home_dir: Optional[Path] = None) -> AuditReport:
    """Run a comprehensive security audit."""
    home = home_dir or Path.home() / ".caveman"
    start = time.monotonic()
    report = AuditReport(scanned_at=time.time())

    # Run all checks
    report.findings.extend(_check_file_permissions(home))
    report.findings.extend(_check_credential_exposure())
    report.findings.extend(_check_config_safety(home))
    report.findings.extend(_check_network_exposure())

    report.duration_ms = (time.monotonic() - start) * 1000
    report.compute_score()
    return report


def _check_file_permissions(home: Path) -> List[AuditFinding]:
    """Check file permissions on sensitive files."""
    findings = []
    sensitive_files = [
        home / "secrets" / "credentials.json",
        home / "config.json",
    ]

    for path in sensitive_files:
        if not path.exists():
            continue
        mode = path.stat().st_mode & 0o777
        if mode & 0o077:  # Group or other can read
            findings.append(AuditFinding(
                category="permissions",
                severity="high",
                title=f"Overly permissive: {path.name}",
                description=f"{path} has mode {oct(mode)} — group/other can access",
                remediation=f"chmod 600 {path}",
                path=str(path),
            ))

    return findings


def _check_credential_exposure() -> List[AuditFinding]:
    """Check for credentials exposed in environment."""
    findings = []
    sensitive_vars = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GITHUB_TOKEN",
        "AWS_SECRET_ACCESS_KEY", "STRIPE_SECRET_KEY",
    ]

    for var in sensitive_vars:
        value = os.environ.get(var, "")
        if value and len(value) < 10:
            findings.append(AuditFinding(
                category="credentials",
                severity="medium",
                title=f"Suspicious {var} value",
                description=f"{var} is set but unusually short ({len(value)} chars)",
            ))

    return findings


def _check_config_safety(home: Path) -> List[AuditFinding]:
    """Check configuration for unsafe settings."""
    findings = []
    config_path = home / "config.json"
    if not config_path.exists():
        return findings

    try:
        import json
        config = json.loads(config_path.read_text(encoding="utf-8"))

        # Check for debug mode in production
        if config.get("debug"):
            findings.append(AuditFinding(
                category="config",
                severity="medium",
                title="Debug mode enabled",
                description="Debug mode may expose sensitive information",
                remediation="Set debug: false in config.json",
            ))

        # Check for wildcard allowlist
        allowlist = config.get("allowlist", [])
        if "*" in str(allowlist):
            findings.append(AuditFinding(
                category="config",
                severity="high",
                title="Wildcard allowlist",
                description="Allowlist contains '*' — anyone can interact",
                remediation="Restrict allowlist to specific users/patterns",
            ))

    except Exception as exc:
        logger.debug("_check_config_safety: suppressed %s", exc)

    return findings


def _check_network_exposure() -> List[AuditFinding]:
    """Check for network exposure."""
    findings = []

    # Check if common ports are listening
    try:
        result = subprocess.run(
            ["lsof", "-i", "-P", "-n"],
            capture_output=True, text=False, timeout=5,
        )
        if result.returncode == 0:
            output = result.stdout.decode("utf-8", errors="replace")
            for line in output.split("\n"):
                if "*:" in line and "LISTEN" in line:
                    # Check for 0.0.0.0 bindings
                    if "0.0.0.0" in line or "*:" in line:
                        port_match = re.search(r":(\d+)", line)
                        if port_match:
                            port = port_match.group(1)
                            if port not in ("22", "443", "80"):
                                findings.append(AuditFinding(
                                    category="network",
                                    severity="low",
                                    title=f"Port {port} listening on all interfaces",
                                    description=line.strip()[:200],
                                ))
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # intentional: FileNotFoundError suppressed

    return findings
