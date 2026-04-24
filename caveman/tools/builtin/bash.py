"""Bash tool — execute shell commands with safety, timeout, and output management.

Ported safety patterns from Hermes shell execution (MIT, Nous Research).

Safety layers:
1. Dangerous pattern blocking (rm -rf /, fork bomb, etc.)
2. File-edit-via-shell guardrail (steer to file_edit/file_write)
3. Self-kill protection (prevent agent from killing its own process tree)
4. Secret scanning (redact API keys from output)
"""
from __future__ import annotations

import asyncio
import os
import re
from typing import Any

from caveman.security.scanner import scan as secret_scan, redact
from caveman.tools.registry import tool

__all__ = [
    "DANGEROUS_PATTERNS",
    "MAX_OUTPUT_CHARS",
    "TRUNCATION_KEEP",
    "bash_exec",
    "register_gateway_pid",
]


# ANSI escape sequence pattern (colors, cursor, etc.)
_ANSI_RE = re.compile(r"\x1b\[[0-9;]*[a-zA-Z]|\x1b\].*?\x07|\x1b\[.*?[@-~]")

# Dangerous command patterns — blocked unconditionally.
# Compiled regexes with word boundaries to prevent false positives
# and catch common evasion techniques.
DANGEROUS_PATTERNS = [
    re.compile(r"\brm\s+-rf\s+/(?:\s|$|\*)"),       # rm -rf / or rm -rf /*
    re.compile(r"\brm\s+-rf\s+~"),                    # rm -rf ~
    re.compile(r"\bdd\s+if=/dev/zero\b"),             # dd if=/dev/zero
    re.compile(r"\bdd\s+if=/dev/random\b"),           # dd if=/dev/random
    re.compile(r"\bmkfs[.\s]"),                        # mkfs. or mkfs<space>
    re.compile(r">\s*/dev/(?:sda|nvme)"),              # > /dev/sda or > /dev/nvme
    re.compile(r":\(\)\s*\{\s*:\|:\s*&\s*\}\s*;"),   # Fork bomb :(){ :|:& };
    re.compile(r"\bchmod\s+-R\s+777\s+/(?:\s|$)"),   # chmod -R 777 /
    re.compile(r"\b(?:curl|wget)\s+.*\|\s*(?:ba)?sh\b"),  # Pipe to shell
    re.compile(r"--no-preserve-root\b"),               # Evasion: --no-preserve-root
    re.compile(r"\$\(.*\)\s*\|\s*(?:ba)?sh\b"),       # Variable expansion to shell: $(...) | sh
    re.compile(r"\bbase64\s+(?:-d|--decode)\b.*\|\s*(?:ba)?sh\b"),  # base64 decode to shell
]

# ── Self-kill protection ──
# Patterns that could kill the gateway process or its ancestors.
# The agent should use /restart (which goes through graceful restart),
# not bash kill commands.

# Gateway PID — set at gateway startup so bash subprocesses know
# which PID to protect even when start_new_session=True severs
# the parent chain.  Populated by register_gateway_pid().
_GATEWAY_PID: int | None = None


def register_gateway_pid(pid: int | None = None) -> None:
    """Record the gateway's own PID for self-kill protection.

    Called once during gateway startup.  If *pid* is None, uses os.getpid().
    """
    global _GATEWAY_PID
    _GATEWAY_PID = pid or os.getpid()
    import logging as _log
    _log.getLogger(__name__).info("Self-kill protection: gateway PID %d registered", _GATEWAY_PID)


# Layer 1: Direct kill commands
_KILL_COMMANDS = re.compile(
    r"""(?:^|[;&|]\s*)       # start of command or chained
    (?:sudo\s+)?             # optional sudo
    (?:kill|pkill|killall)   # kill family
    \b""",
    re.VERBOSE | re.IGNORECASE,
)

# Layer 2: Indirect kill via subshell, xargs, python, etc.
_INDIRECT_KILL_PATTERNS = re.compile(
    r"""
    (?:xargs\s+(?:.*\s)?kill\b)                     |  # pgrep | xargs kill
    (?:bash\s+-c\s+[\"'].*kill\b)                    |  # bash -c "kill ..."
    (?:sh\s+-c\s+[\"'].*kill\b)                      |  # sh -c "kill ..."
    (?:python[3]?\s+-c\s+.*os\.kill\b)               |  # python -c "os.kill(...)"
    (?:python[3]?\s+-c\s+.*signal\.SIGKILL\b)        |  # python -c "signal..."
    (?:python[3]?\s+-c\s+.*signal\.SIGTERM\b)        |  # python -c "signal..."
    (?:perl\s+-e\s+.*kill\b)                         |  # perl -e "kill ..."
    (?:ruby\s+-e\s+.*Process\.kill\b)                   # ruby -e "Process.kill..."
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Layer 3: Commands that target caveman processes by name (always dangerous)
_CAVEMAN_PROCESS_KILL = re.compile(
    r"""
    (?:
        (?:pkill|killall).*(?:caveman|run_gateway|gateway_lifecycle)  |  # pkill/killall caveman
        (?:caveman|run_gateway|gateway_lifecycle).*(?:pkill|killall)  |  # unlikely but defensive
        pgrep\s+.*(?:caveman|run_gateway|gateway_lifecycle).*\|\s*(?:xargs\s+)?kill  # pgrep caveman | xargs kill
    )
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Extract numeric PIDs from kill commands (e.g., "kill 12345" or "kill -9 12345")
_KILL_PID_PATTERN = re.compile(
    r"""(?:^|[;&|]\s*)
    (?:sudo\s+)?
    kill\s+
    (?:-\w+\s+)*             # optional signal flags (-9, -TERM, etc.)
    ([\d\s]+)                # one or more PIDs
    """,
    re.VERBOSE | re.IGNORECASE,
)

# Extract process name patterns from pkill/killall
_PKILL_PATTERN = re.compile(
    r"""(?:^|[;&|]\s*)
    (?:sudo\s+)?
    (?:pkill|killall)\s+
    (?:-\w+\s+)*             # optional flags
    [\"']?([^\"';&|]+)       # process name/pattern
    """,
    re.VERBOSE | re.IGNORECASE,
)


def _detect_gateway_pid() -> int | None:
    """Fallback: detect gateway PID from PID file."""
    try:
        from caveman.gateway.status import get_running_pid
        return get_running_pid()
    except Exception:  # intentional — status module may not be available
        pass
    return None


def _get_process_tree_pids() -> set[int]:
    """Get PIDs of the current process and all ancestors up to init.

    This covers: the gateway process itself, the Python interpreter,
    and any parent shell that launched it.
    """
    pids = set()
    try:
        pid = os.getpid()
        pids.add(pid)
        # Walk up the process tree
        while pid > 1:
            try:
                pid = os.getppid() if pid == os.getpid() else _get_ppid(pid)
                if pid <= 1:
                    break
                pids.add(pid)
            except (ProcessLookupError, PermissionError, OSError):
                break
    except Exception:
        # At minimum, protect our own PID
        pids.add(os.getpid())
    # Always include the gateway PID — bash runs in a new session
    # (start_new_session=True) so the parent walk may not reach it.
    if _GATEWAY_PID is not None:
        pids.add(_GATEWAY_PID)
    else:
        # Fallback: read PID file or detect via pgrep
        gw_pid = _detect_gateway_pid()
        if gw_pid:
            pids.add(gw_pid)
    return pids


def _get_ppid(pid: int) -> int:
    """Get parent PID of a given process (macOS/Linux)."""
    import subprocess
    try:
        result = subprocess.run(
            ["ps", "-o", "ppid=", "-p", str(pid)],
            capture_output=True, text=True, timeout=2,
        )
        return int(result.stdout.strip())
    except (subprocess.TimeoutExpired, ValueError, OSError):
        return 0


def _is_self_kill(command: str) -> str | None:
    """Detect if a command would kill the gateway's own process tree.

    Three layers of detection:
    1. Direct kill with PID matching our process tree
    2. Indirect kill via subshell/xargs/python/perl/ruby
    3. Kill-by-name targeting caveman processes
    4. Variable/subshell expansion that references caveman or self

    Returns a description of the blocked action, or None if safe.
    """
    cmd_lower = command.lower()

    # Layer 3: Kill-by-name targeting caveman (always block, no PID check needed)
    if _CAVEMAN_PROCESS_KILL.search(command):
        return "command targets caveman processes by name"

    # Layer 2: Indirect kill patterns (subshell, xargs, python -c, etc.)
    if _INDIRECT_KILL_PATTERNS.search(command):
        # Check if it references caveman/gateway/self
        if any(kw in cmd_lower for kw in ("caveman", "gateway", "run_gateway",
                                           "getppid", "getpid", "self", "/proc/self")):
            return "indirect kill targeting gateway process"

    # Layer 4: Kill with unresolvable PID that references caveman/self
    # Catches: "PID=$(pgrep caveman); kill $PID" and "kill $(cat /proc/self/ppid)"
    if _KILL_COMMANDS.search(command):
        # If the command contains both a kill AND a reference to caveman/self,
        # it's likely trying to kill the gateway via variable expansion
        self_refs = ("caveman", "gateway", "run_gateway", "/proc/self",
                     "getppid", "getpid", "$$")
        if any(ref in cmd_lower for ref in self_refs):
            return "kill command with caveman/self reference in same pipeline"

    # Layer 1: Direct kill with specific PIDs
    for match in _KILL_PID_PATTERN.finditer(command):
        pid_str = match.group(1).strip()
        try:
            target_pids = {int(p) for p in pid_str.split() if p.strip().isdigit()}
        except ValueError:
            continue
        protected = _get_process_tree_pids()
        overlap = target_pids & protected
        if overlap:
            return f"kill targeting own process tree (PIDs: {overlap})"

    # Layer 1b: pkill/killall targeting caveman-related process names
    for match in _PKILL_PATTERN.finditer(command):
        pattern = match.group(1).strip().lower()
        dangerous_patterns = ("caveman", "python.*caveman", "gateway", "run_gateway")
        for dp in dangerous_patterns:
            if dp in pattern or re.search(dp, pattern):
                return f"pkill/killall targeting '{pattern}' (matches gateway process)"

    return None

# Max output size before truncation (chars)
MAX_OUTPUT_CHARS = 100_000
TRUNCATION_KEEP = 2_000  # Keep first/last N chars when truncating


def _is_dangerous(command: str) -> str | None:
    """Check if command matches dangerous patterns. Returns pattern or None."""
    cmd_lower = command.lower().strip()
    for pattern in DANGEROUS_PATTERNS:
        if pattern.search(cmd_lower):
            return pattern.pattern
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
    """Truncate long output, keeping head and tail. Strips ANSI escape codes."""
    # Strip ANSI escape sequences (colors, cursor movement, etc.)
    text = _ANSI_RE.sub("", text)
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
        "timeout": {"type": "integer", "description": "Timeout in seconds (default 120, max 600)"},
        "cwd": {"type": "string", "description": "Working directory"},
    },
    required=["command"],
)
async def bash_exec(
    command: str,
    timeout: int = 120,
    cwd: str | None = None,
) -> dict[str, Any]:
    """Execute bash command with safety checks and output management."""
    # Safety check 1: dangerous patterns
    danger = _is_dangerous(command)
    if danger:
        return {
            "stdout": "",
            "stderr": f"⚠️ Potentially dangerous command (pattern: '{danger}'). "
                      f"Use the clarify tool to ask the user for permission before retrying.",
            "returncode": -1,
            "success": False,
            "needs_approval": True,
            "blocked_pattern": danger,
        }

    # Safety check 2: self-kill protection (3 layers)
    if _KILL_COMMANDS.search(command) or _INDIRECT_KILL_PATTERNS.search(command) or _CAVEMAN_PROCESS_KILL.search(command):
        self_kill = _is_self_kill(command)
        if self_kill:
            return {
                "stdout": "",
                "stderr": (
                    f"⛔ Blocked: {self_kill}. "
                    "Use the /restart command for graceful gateway restart. "
                    "Killing your own process will cause an unrecoverable crash."
                ),
                "returncode": -1,
                "success": False,
            }

    # Safety check 3: file-edit guardrail
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
    timeout = max(1, min(timeout, 600))

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
        # Build clean env (foreign agent vars already stripped at startup)
        from caveman.runtime_identity import build_clean_env
        env = build_clean_env()

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
