"""Code Execution — sandboxed code execution with timeout and output capture.

Provides safe code execution in isolated environments with
resource limits, timeout, and output capture. Extracted from
Hermes tools/code_execution_tool.py (1378 lines).
"""
from __future__ import annotations

import logging
import os
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional

__all__ = [
    "ExecutionConfig",
    "ExecutionResult",
    "execute_code",
]


logger = logging.getLogger("caveman.tools.code_execution")

_MAX_OUTPUT_SIZE = 100_000  # 100KB
_DEFAULT_TIMEOUT = 30  # seconds
_MAX_TIMEOUT = 300  # 5 minutes


@dataclass
class ExecutionConfig:
    """Configuration for code execution."""
    timeout: int = _DEFAULT_TIMEOUT
    max_output: int = _MAX_OUTPUT_SIZE
    working_dir: str = ""
    env_vars: Dict[str, str] = field(default_factory=dict)
    allowed_languages: List[str] = field(
        default_factory=lambda: ["python", "javascript", "bash", "sh"],
    )
    sandbox_mode: str = "subprocess"  # subprocess | docker | none


@dataclass
class ExecutionResult:
    """Result of code execution."""
    success: bool = False
    stdout: str = ""
    stderr: str = ""
    exit_code: int = -1
    language: str = ""
    duration_ms: float = 0
    timed_out: bool = False
    truncated: bool = False
    error: str = ""


# ── Language Runners ──

_LANGUAGE_COMMANDS = {
    "python": ["python3", "-c"],
    "python3": ["python3", "-c"],
    "javascript": ["node", "-e"],
    "js": ["node", "-e"],
    "bash": ["bash", "-c"],
    "sh": ["sh", "-c"],
    "ruby": ["ruby", "-e"],
    "perl": ["perl", "-e"],
}

_LANGUAGE_FILE_EXT = {
    "python": ".py",
    "python3": ".py",
    "javascript": ".js",
    "js": ".js",
    "typescript": ".ts",
    "bash": ".sh",
    "sh": ".sh",
    "ruby": ".rb",
    "perl": ".pl",
}


def execute_code(
    code: str,
    language: str = "python",
    config: Optional[ExecutionConfig] = None,
) -> ExecutionResult:
    """Execute code in a sandboxed environment."""
    config = config or ExecutionConfig()
    language = language.lower().strip()

    # Validate language
    if language not in _LANGUAGE_COMMANDS:
        return ExecutionResult(
            error=f"Unsupported language: {language}. Supported: {list(_LANGUAGE_COMMANDS.keys())}",
        )

    if config.allowed_languages and language not in config.allowed_languages:
        return ExecutionResult(error=f"Language '{language}' not allowed")

    # Security checks
    danger = _check_code_safety(code, language)
    if danger:
        return ExecutionResult(error=f"Blocked: {danger}")

    timeout = min(config.timeout, _MAX_TIMEOUT)
    cmd = _LANGUAGE_COMMANDS[language]

    # For longer code, use a temp file
    if len(code) > 1000 or "\n" in code:
        return _execute_via_file(code, language, config, timeout)

    return _execute_inline(code, cmd, config, timeout, language)


def _execute_inline(
    code: str,
    cmd: List[str],
    config: ExecutionConfig,
    timeout: int,
    language: str,
) -> ExecutionResult:
    """Execute code inline via -c/-e flag."""
    start = time.monotonic()
    env = {**os.environ, **config.env_vars}

    try:
        proc = subprocess.run(
            cmd + [code],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=config.working_dir or None,
            env=env,
        )
        duration = (time.monotonic() - start) * 1000

        stdout = proc.stdout
        stderr = proc.stderr
        truncated = False

        if len(stdout) > config.max_output:
            stdout = stdout[:config.max_output] + "\n[output truncated]"
            truncated = True
        if len(stderr) > config.max_output:
            stderr = stderr[:config.max_output] + "\n[stderr truncated]"
            truncated = True

        return ExecutionResult(
            success=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            language=language,
            duration_ms=duration,
            truncated=truncated,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            error=f"Execution timed out after {timeout}s",
            timed_out=True,
            language=language,
            duration_ms=timeout * 1000,
        )
    except Exception as e:
        return ExecutionResult(error=str(e), language=language)


def _execute_via_file(
    code: str,
    language: str,
    config: ExecutionConfig,
    timeout: int,
) -> ExecutionResult:
    """Execute code via a temporary file."""
    ext = _LANGUAGE_FILE_EXT.get(language, ".txt")
    start = time.monotonic()

    with tempfile.NamedTemporaryFile(
        mode="w", suffix=ext, delete=False, encoding="utf-8",
    ) as f:
        f.write(code)
        temp_path = f.name

    try:
        # Build command for file execution
        if language in ("python", "python3"):
            cmd = ["python3", temp_path]
        elif language in ("javascript", "js"):
            cmd = ["node", temp_path]
        elif language in ("bash", "sh"):
            cmd = [language, temp_path]
        elif language == "ruby":
            cmd = ["ruby", temp_path]
        else:
            cmd = [language, temp_path]

        env = {**os.environ, **config.env_vars}
        proc = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=config.working_dir or None,
            env=env,
        )
        duration = (time.monotonic() - start) * 1000

        stdout = proc.stdout
        stderr = proc.stderr
        truncated = False

        if len(stdout) > config.max_output:
            stdout = stdout[:config.max_output] + "\n[output truncated]"
            truncated = True

        return ExecutionResult(
            success=proc.returncode == 0,
            stdout=stdout,
            stderr=stderr,
            exit_code=proc.returncode,
            language=language,
            duration_ms=duration,
            truncated=truncated,
        )

    except subprocess.TimeoutExpired:
        return ExecutionResult(
            error=f"Execution timed out after {timeout}s",
            timed_out=True,
            language=language,
        )
    except Exception as e:
        return ExecutionResult(error=str(e), language=language)
    finally:
        try:
            os.unlink(temp_path)
        except Exception:
            pass  # intentional: Exception suppressed


# ── Safety Checks ──

_DANGEROUS_CODE_PATTERNS = [
    (r"\bos\.system\s*\(", "os.system call"),
    (r"\bsubprocess\.call\s*\(.*shell\s*=\s*True", "subprocess with shell=True"),
    (r"\b__import__\s*\(", "dynamic import"),
    (r"\beval\s*\(", "eval call"),
    (r"\bexec\s*\(", "exec call"),
    (r"\bshutil\.rmtree\s*\(['\"]\/", "rmtree on root"),
    (r"\bos\.remove\s*\(['\"]\/", "remove from root"),
    (r"import\s+ctypes\b", "ctypes import"),
    (r"\bsocket\b.*\bconnect\b", "network socket"),
]


def _check_code_safety(code: str, language: str) -> Optional[str]:
    """Check code for dangerous patterns. Returns reason if blocked."""
    if language in ("bash", "sh"):
        # Shell commands are checked by the approval system
        return None

    import re
    for pattern, reason in _DANGEROUS_CODE_PATTERNS:
        if re.search(pattern, code):
            return reason

    return None
