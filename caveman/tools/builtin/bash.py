"""Bash tool — execute shell commands with safety, timeout, and output management.

Ported safety patterns from Hermes shell execution (MIT, Nous Research).
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from caveman.security.scanner import scan as secret_scan, redact
from caveman.tools.registry import tool

# Dangerous command patterns — blocked unconditionally
DANGEROUS_PATTERNS = [
    "rm -rf /", "rm -rf ~", "rm -rf /*",
    "dd if=/dev/zero", "dd if=/dev/random",
    "mkfs.", "mkfs ",
    "> /dev/sda", "> /dev/nvme",
    ":(){ :|:& };",  # Fork bomb
    "chmod -R 777 /",
    "curl.*|.*sh", "wget.*|.*sh",  # Pipe to shell
]

# Max output size before truncation (chars)
MAX_OUTPUT_CHARS = 100_000
TRUNCATION_KEEP = 2_000  # Keep first/last N chars when truncating


def _is_dangerous(command: str) -> str | None:
    """Check if command matches dangerous patterns. Returns pattern or None."""
    cmd_lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern in cmd_lower:
            return pattern
    # Check for pipe-to-shell patterns
    if re.search(r"(curl|wget)\s+.*\|\s*(ba)?sh", cmd_lower):
        return "pipe to shell"
    return None


# File-edit-via-shell patterns — should use file_edit/file_write instead
_FILE_EDIT_PATTERNS = [
    r"echo\s+['\"].*['\"]\s*>\s*\S+",       # echo "..." > file
    r"cat\s*>\s*\S+\s*<<",                    # cat > file << EOF
    r"sed\s+-i",                               # sed -i
    r"perl\s+-[pi]",                           # perl -pi
    r"python[3]?\s+-c\s+.*open\(",            # python -c "open(...).write(...)"
    r"printf\s+.*>\s*\S+",                     # printf ... > file
    r"tee\s+\S+",                              # tee file (when used for writing)
]


def _is_file_edit_via_shell(command: str) -> str | None:
    """Detect shell commands that should use file_edit/file_write instead."""
    cmd = command.strip()
    for pattern in _FILE_EDIT_PATTERNS:
        if re.search(pattern, cmd, re.IGNORECASE):
            return pattern
    return None


def _truncate_output(text: str) -> str:
    """Truncate long output, keeping head and tail."""
    if len(text) <= MAX_OUTPUT_CHARS:
        return text
    head = text[:TRUNCATION_KEEP]
    tail = text[-TRUNCATION_KEEP:]
    skipped = len(text) - 2 * TRUNCATION_KEEP
    return f"{head}\n\n... [{skipped:,} chars truncated] ...\n\n{tail}"


@tool(
    name="bash",
    description="Execute a bash command. Returns stdout, stderr, return code.",
    params={
        "command": {"type": "string", "description": "Bash command to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 30, max 300)"},
        "cwd": {"type": "string", "description": "Working directory"},
    },
    required=["command"],
)
async def bash_exec(
    command: str,
    timeout: int = 30,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Execute bash command with safety checks and output management."""
    # Safety check
    danger = _is_dangerous(command)
    if danger:
        return {
            "stdout": "",
            "stderr": f"⛔ Blocked dangerous pattern: '{danger}'",
            "returncode": -1,
            "success": False,
        }

    # File-edit guardrail: steer toward file_edit/file_write
    file_edit_match = _is_file_edit_via_shell(command)
    if file_edit_match:
        return {
            "stdout": "",
            "stderr": (
                "⚠️ This looks like a file edit via shell. "
                "Use file_edit or file_write instead — they are safer, "
                "atomic, and produce better diffs. "
                "Bash is for running commands, tests, git, and system inspection."
            ),
            "returncode": -1,
            "error": "Use file_edit/file_write for source code changes",
            "success": False,
        }

    # Clamp timeout
    timeout = max(1, min(timeout, 300))

    # Resolve working directory
    work_dir = cwd or os.getcwd()
    if not os.path.isdir(work_dir):
        return {
            "stdout": "",
            "stderr": f"Working directory not found: {work_dir}",
            "returncode": -1,
            "success": False,
        }

    try:
        # Build env with venv PATH if running inside one
        env = {**os.environ, "TERM": "dumb", "NO_COLOR": "1"}
        venv_bin = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))), ".venv", "bin")
        if os.path.isdir(venv_bin):
            env["PATH"] = venv_bin + ":" + env.get("PATH", "")
            env["VIRTUAL_ENV"] = os.path.dirname(venv_bin)

        proc = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=work_dir,
            env=env,
            start_new_session=True,  # Isolate process group for clean kill
        )
        try:
            stdout_bytes, stderr_bytes = await asyncio.wait_for(
                proc.communicate(), timeout=timeout,
            )
        except asyncio.TimeoutError:
            # Kill entire process group, not just the shell leader
            import os as _os, signal as _sig
            try:
                pgid = _os.getpgid(proc.pid)
                _os.killpg(pgid, _sig.SIGKILL)
            except (ProcessLookupError, PermissionError, OSError):
                proc.kill()
            await proc.communicate()
            return {
                "stdout": "",
                "stderr": f"⏱️ Timed out after {timeout}s. Consider increasing timeout.",
                "returncode": -1,
                "success": False,
            }

        out = stdout_bytes.decode("utf-8", errors="replace")
        err = stderr_bytes.decode("utf-8", errors="replace")

        # Secret scanning
        scan_result = secret_scan(out + err)
        if scan_result.has_secrets:
            out, err = redact(out), redact(err)

        # Truncate long output
        out = _truncate_output(out)
        err = _truncate_output(err)

        return {
            "stdout": out,
            "stderr": err,
            "returncode": proc.returncode,
            "success": proc.returncode == 0,
        }
    except Exception as e:
        return {
            "stdout": "",
            "stderr": f"Execution error: {e}",
            "returncode": -1,
            "success": False,
        }
