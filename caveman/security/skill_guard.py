"""Skill Guard — security scanner for externally-sourced skills.

Ported from Hermes skills_guard.py (MIT, Nous Research) with Caveman adaptations.
Scans skill files for dangerous patterns before installation.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

__all__ = ["scan_skill", "should_allow_install", "ScanResult", "TrustLevel",
    "format_scan_report"]


class TrustLevel:
    """Trust tier for skill execution — controls sandbox and capability grants."""
    BUILTIN = "builtin"    # Ships with Caveman, never scanned
    TRUSTED = "trusted"    # Official registry, caution allowed
    COMMUNITY = "community"  # Everything else, strict


@dataclass
class Finding:
    """A single security finding."""
    severity: str  # critical, warning, caution
    pattern: str
    file: str
    line: int
    context: str


@dataclass
class ScanResult:
    """Result of scanning a skill."""
    skill_name: str
    trust_level: str
    findings: list[Finding] = field(default_factory=list)
    files_scanned: int = 0
    hash: str = ""

    @property
    def verdict(self) -> str:
        if any(f.severity == "critical" for f in self.findings):
            return "blocked"
        if any(f.severity in ("warning", "caution") for f in self.findings):
            return "caution"
        return "clean"


# --- Dangerous patterns ---

_CRITICAL_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Data exfiltration
    ("data_exfil", re.compile(
        r"(requests\.post|urllib\.request\.urlopen|httpx\.post|aiohttp\..*\.post)"
        r".*(?:api_key|token|secret|password|credential)",
        re.IGNORECASE,
    )),
    # Prompt injection
    ("prompt_inject", re.compile(
        r"(ignore.*previous.*instructions|you are now|system prompt override|forget.*rules)",
        re.IGNORECASE,
    )),
    # Destructive commands
    ("destructive_cmd", re.compile(
        r"(rm\s+-rf\s+/|shutil\.rmtree.*(/|home)|os\.remove.*\*)",
        re.IGNORECASE,
    )),
    # Persistence / backdoor
    ("persistence", re.compile(
        r"(crontab|launchd|systemd|at\s+|schtasks|startup.*folder)",
        re.IGNORECASE,
    )),
    # Crypto mining
    ("crypto_mine", re.compile(
        r"(xmrig|minerd|cryptonight|stratum\+tcp)",
        re.IGNORECASE,
    )),
    # Reverse shell
    ("reverse_shell", re.compile(
        r"(socket\.connect|nc\s+-e|bash\s+-i\s+>&|/dev/tcp/)",
        re.IGNORECASE,
    )),
]

_WARNING_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Network access
    ("network_access", re.compile(
        r"(requests\.(get|post)|urllib|httpx|aiohttp|socket\.socket)",
        re.IGNORECASE,
    )),
    # File system writes outside workspace
    ("fs_write_outside", re.compile(
        r"(open\(.*['\"]w['\"]|Path.*write_text|shutil\.copy).*(/etc|/usr|/var|/tmp)",
        re.IGNORECASE,
    )),
    # Subprocess execution
    ("subprocess", re.compile(
        r"(subprocess\.(run|Popen|call)|os\.system|os\.popen)",
        re.IGNORECASE,
    )),
    # Environment variable access
    ("env_access", re.compile(
        r"os\.environ\[.*(?:KEY|TOKEN|SECRET|PASSWORD)",
        re.IGNORECASE,
    )),
]

_CAUTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    # Dynamic code execution
    ("dynamic_exec", re.compile(
        r"(exec\(|eval\(|compile\(|__import__)",
    )),
    # Pickle (deserialization attack vector)
    ("pickle", re.compile(
        r"(pickle\.loads?|cloudpickle|dill\.loads?)",
    )),
]

_SCAN_EXTENSIONS = {".py", ".sh", ".bash", ".js", ".ts", ".yaml", ".yml", ".md"}


def _hash_directory(path: Path) -> str:
    """SHA256 hash of all files in a directory."""
    h = hashlib.sha256()
    for f in sorted(path.rglob("*")):
        if f.is_file() and f.suffix in _SCAN_EXTENSIONS:
            h.update(f.read_bytes())
    return h.hexdigest()[:16]


def scan_skill(skill_path: Path, trust_level: str = TrustLevel.COMMUNITY) -> ScanResult:
    """Scan a skill directory for security issues."""
    result = ScanResult(
        skill_name=skill_path.name,
        trust_level=trust_level,
        hash=_hash_directory(skill_path) if skill_path.is_dir() else "",
    )

    if trust_level == TrustLevel.BUILTIN:
        return result  # Never scan builtins

    files = []
    if skill_path.is_dir():
        files = [f for f in skill_path.rglob("*") if f.is_file() and f.suffix in _SCAN_EXTENSIONS]
    elif skill_path.is_file():
        files = [skill_path]

    result.files_scanned = len(files)

    for fpath in files:
        try:
            content = fpath.read_text(encoding="utf-8", errors="replace")
        except Exception as e:
            logger.debug("suppressed: %s", e)
            continue

        rel = str(fpath.relative_to(skill_path)) if skill_path.is_dir() else fpath.name

        for line_no, line in enumerate(content.splitlines(), 1):
            # Skip comments
            stripped = line.strip()
            if stripped.startswith("#") or stripped.startswith("//"):
                continue

            for pattern_name, pattern in _CRITICAL_PATTERNS:
                if pattern.search(line):
                    result.findings.append(Finding(
                        severity="critical", pattern=pattern_name,
                        file=rel, line=line_no, context=line.strip()[:120],
                    ))

            for pattern_name, pattern in _WARNING_PATTERNS:
                if pattern.search(line):
                    result.findings.append(Finding(
                        severity="warning", pattern=pattern_name,
                        file=rel, line=line_no, context=line.strip()[:120],
                    ))

            for pattern_name, pattern in _CAUTION_PATTERNS:
                if pattern.search(line):
                    result.findings.append(Finding(
                        severity="caution", pattern=pattern_name,
                        file=rel, line=line_no, context=line.strip()[:120],
                    ))

    return result


def should_allow_install(result: ScanResult) -> tuple[bool, str]:
    """Determine if a skill should be allowed based on scan results."""
    if result.trust_level == TrustLevel.BUILTIN:
        return True, "builtin skill"

    if result.verdict == "blocked":
        critical = [f for f in result.findings if f.severity == "critical"]
        return False, f"{len(critical)} critical finding(s): {', '.join(f.pattern for f in critical[:3])}"

    if result.verdict == "caution":
        if result.trust_level == TrustLevel.TRUSTED:
            return True, "trusted source with caution findings"
        warnings = [f for f in result.findings if f.severity in ("warning", "caution")]
        return False, f"{len(warnings)} finding(s) from community source: {', '.join(f.pattern for f in warnings[:3])}"

    return True, "clean scan"


def format_scan_report(result: ScanResult) -> str:
    """Format scan results as human-readable report."""
    lines = [
        f"🔍 Skill Guard Report: {result.skill_name}",
        f"   Trust: {result.trust_level} | Files: {result.files_scanned} | Hash: {result.hash}",
        f"   Verdict: {result.verdict.upper()}",
    ]
    if result.findings:
        lines.append("")
        for f in result.findings:
            icon = {"critical": "🔴", "warning": "🟡", "caution": "🟠"}.get(f.severity, "⚪")
            lines.append(f"   {icon} [{f.severity}] {f.pattern} — {f.file}:{f.line}")
            lines.append(f"      {f.context}")
    return "\n".join(lines)
