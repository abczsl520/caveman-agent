"""Skills Guard — security validation for skill bundles.

Validates skill bundles before installation, checking for
dangerous patterns, excessive permissions, and policy violations.
Extracted from Hermes tools/skills_guard.py (977 lines).
"""
from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

__all__ = [
    "ThreatPattern",
    "THREAT_PATTERNS",
    "ScanResult",
    "scan_bundle",
    "quarantine_skill",
    "list_quarantined",
    "log_audit",
    "guard_install",
]


logger = logging.getLogger("caveman.tools.skills_guard")

_QUARANTINE_DIR = Path.home() / ".caveman" / "quarantine"
_AUDIT_LOG = Path.home() / ".caveman" / "skills_audit.jsonl"


# ── Threat Patterns ──

@dataclass
class ThreatPattern:
    """A pattern that indicates a potential threat."""
    name: str
    pattern: re.Pattern
    severity: str = "warning"  # info | warning | critical
    description: str = ""


THREAT_PATTERNS = [
    ThreatPattern("subprocess", re.compile(r"subprocess\.(run|call|Popen|check_output)"), "critical", "Process execution"),
    ThreatPattern("os_system", re.compile(r"os\.system\s*\("), "critical", "Shell command execution"),
    ThreatPattern("eval", re.compile(r"\beval\s*\("), "critical", "Dynamic code evaluation"),
    ThreatPattern("exec", re.compile(r"\bexec\s*\("), "critical", "Dynamic code execution"),
    ThreatPattern("__import__", re.compile(r"__import__\s*\("), "critical", "Dynamic import"),
    ThreatPattern("compile", re.compile(r"\bcompile\s*\("), "warning", "Code compilation"),
    ThreatPattern("ctypes", re.compile(r"import\s+ctypes"), "critical", "Native code access"),
    ThreatPattern("socket", re.compile(r"import\s+socket"), "warning", "Network access"),
    ThreatPattern("requests", re.compile(r"requests?\.(get|post|put|delete|patch)\s*\("), "warning", "HTTP requests"),
    ThreatPattern("urllib", re.compile(r"urllib\.request\.urlopen"), "warning", "URL access"),
    ThreatPattern("file_write", re.compile(r"open\s*\([^)]*['\"]w['\"]"), "warning", "File write"),
    ThreatPattern("rmtree", re.compile(r"shutil\.rmtree"), "critical", "Recursive deletion"),
    ThreatPattern("env_access", re.compile(r"os\.environ"), "warning", "Environment variable access"),
    ThreatPattern("path_traversal", re.compile(r"\.\./"), "warning", "Path traversal"),
    ThreatPattern("base64_decode", re.compile(r"base64\.(b64decode|decodebytes)"), "warning", "Base64 decoding (possible obfuscation)"),
]


@dataclass
class ScanResult:
    """Result of a security scan."""
    passed: bool = True
    findings: List[Dict[str, Any]] = field(default_factory=list)
    critical_count: int = 0
    warning_count: int = 0
    info_count: int = 0
    files_scanned: int = 0
    scan_duration_ms: float = 0


def scan_bundle(files: Dict[str, str]) -> ScanResult:
    """Scan a skill bundle for security threats."""
    start = time.monotonic()
    result = ScanResult(files_scanned=len(files))

    for filepath, content in files.items():
        for i, line in enumerate(content.split("\n"), 1):
            for threat in THREAT_PATTERNS:
                if threat.pattern.search(line):
                    finding = {
                        "file": filepath,
                        "line": i,
                        "threat": threat.name,
                        "severity": threat.severity,
                        "description": threat.description,
                        "content": line.strip()[:200],
                    }
                    result.findings.append(finding)
                    if threat.severity == "critical":
                        result.critical_count += 1
                    elif threat.severity == "warning":
                        result.warning_count += 1
                    else:
                        result.info_count += 1

    result.passed = result.critical_count == 0
    result.scan_duration_ms = (time.monotonic() - start) * 1000
    return result


# ── Quarantine ──

def quarantine_skill(
    skill_name: str,
    files: Dict[str, str],
    reason: str,
) -> Path:
    """Move a skill to quarantine."""
    _QUARANTINE_DIR.mkdir(parents=True, exist_ok=True)
    quarantine_path = _QUARANTINE_DIR / f"{skill_name}_{int(time.time())}"
    quarantine_path.mkdir(parents=True, exist_ok=True)

    for filepath, content in files.items():
        file_path = quarantine_path / filepath
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(content, encoding="utf-8")

    # Write quarantine metadata
    meta = {
        "skill_name": skill_name,
        "reason": reason,
        "quarantined_at": time.time(),
        "files": list(files.keys()),
    }
    (quarantine_path / "_quarantine.json").write_text(
        json.dumps(meta, ensure_ascii=False), encoding="utf-8",
    )

    return quarantine_path


def list_quarantined() -> List[Dict[str, Any]]:
    """List quarantined skills."""
    if not _QUARANTINE_DIR.exists():
        return []
    results = []
    for d in _QUARANTINE_DIR.iterdir():
        if not d.is_dir():
            continue
        meta_path = d / "_quarantine.json"
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
                results.append(meta)
            except Exception:
                results.append({"skill_name": d.name, "path": str(d)})
    return results


# ── Audit Log ──

def log_audit(
    action: str,
    skill_name: str,
    result: Optional[ScanResult] = None,
    details: Optional[Dict[str, Any]] = None,
) -> None:
    """Log a security audit event."""
    entry = {
        "timestamp": time.time(),
        "action": action,
        "skill_name": skill_name,
    }
    if result:
        entry["passed"] = result.passed
        entry["critical"] = result.critical_count
        entry["warnings"] = result.warning_count
        entry["findings"] = len(result.findings)
    if details:
        entry.update(details)

    try:
        _AUDIT_LOG.parent.mkdir(parents=True, exist_ok=True)
        with open(_AUDIT_LOG, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")
    except Exception as e:
        logger.debug("Failed to write audit log: %s", e)


# ── Install Guard ──

def guard_install(
    skill_name: str,
    files: Dict[str, str],
    auto_quarantine: bool = True,
) -> Dict[str, Any]:
    """Guard a skill installation — scan, quarantine if dangerous."""
    scan = scan_bundle(files)
    log_audit("install_scan", skill_name, scan)

    if scan.passed:
        return {
            "allowed": True,
            "scan": scan,
        }

    if auto_quarantine:
        quarantine_path = quarantine_skill(
            skill_name, files,
            reason=f"{scan.critical_count} critical findings",
        )
        log_audit("quarantined", skill_name, scan, {"path": str(quarantine_path)})
        return {
            "allowed": False,
            "reason": f"{scan.critical_count} critical security findings",
            "quarantine_path": str(quarantine_path),
            "scan": scan,
        }

    return {
        "allowed": False,
        "reason": f"{scan.critical_count} critical security findings",
        "scan": scan,
    }
