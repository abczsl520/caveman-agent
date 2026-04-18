"""Sandbox tool — safe Python code execution in isolated subprocess."""
from __future__ import annotations

import ast
import asyncio
import logging
import os
import signal
import tempfile
import time

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

_MAX_TIMEOUT = 60
_MAX_OUTPUT = 100 * 1024  # 100KB


def _minimal_env(sandbox_dir: str) -> dict[str, str]:
    """Build minimal env — don't leak host API keys into sandbox."""
    return {
        "HOME": sandbox_dir,
        "TMPDIR": sandbox_dir,
        "PATH": "/usr/local/bin:/usr/bin:/bin",
        "LANG": "en_US.UTF-8",
        "PYTHONDONTWRITEBYTECODE": "1",
        "CAVEMAN_SANDBOX": "1",
    }


async def _kill_process_group(proc: asyncio.subprocess.Process) -> None:
    """Kill entire process group (not just leader) to prevent orphans."""
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGTERM)
    except (ProcessLookupError, PermissionError, OSError):
        pass
    await asyncio.sleep(0.5)
    try:
        pgid = os.getpgid(proc.pid)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError, OSError):
        try:
            proc.kill()
        except (ProcessLookupError, OSError):
            pass
    try:
        await asyncio.wait_for(proc.communicate(), timeout=3)
    except (asyncio.TimeoutError, ProcessLookupError, OSError):
        pass


@tool(
    name="sandbox_exec",
    description="Execute Python code safely in an isolated subprocess",
    params={
        "code": {"type": "string", "description": "Python code to execute"},
        "timeout": {"type": "integer", "description": "Timeout in seconds (max 60)", "default": 10},
    },
    required=["code"],
)
async def sandbox_exec(code: str, timeout: int = 10) -> dict:
    """Execute Python code in a sandboxed subprocess."""
    timeout = min(max(1, timeout), _MAX_TIMEOUT)
    tmp_dir = tempfile.mkdtemp(prefix="caveman_sandbox_")
    tmp_file = os.path.join(tmp_dir, "main.py")
    try:
        with open(tmp_file, "w", encoding="utf-8") as f:
            f.write(code)

        env = _minimal_env(tmp_dir)
        start = time.monotonic()
        proc = await asyncio.create_subprocess_exec(
            "python3", "-u", tmp_file,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=tmp_dir,
            env=env,
            start_new_session=True,  # New session for clean group kill
        )
        try:
            stdout_b, stderr_b = await asyncio.wait_for(proc.communicate(), timeout=timeout)
        except asyncio.TimeoutError:
            await _kill_process_group(proc)
            return {"ok": False, "stdout": "", "stderr": "Timeout exceeded", "returncode": -1, "duration_s": round(time.monotonic() - start, 3)}

        duration = round(time.monotonic() - start, 3)
        stdout = stdout_b[:_MAX_OUTPUT].decode("utf-8", errors="replace")
        stderr = stderr_b[:_MAX_OUTPUT].decode("utf-8", errors="replace")
        return {
            "ok": proc.returncode == 0,
            "stdout": stdout,
            "stderr": stderr,
            "returncode": proc.returncode,
            "duration_s": duration,
        }
    except Exception as e:
        return {"ok": False, "stdout": "", "stderr": str(e), "returncode": -1, "duration_s": 0.0}
    finally:
        import shutil
        shutil.rmtree(tmp_dir, ignore_errors=True)


@tool(
    name="sandbox_eval",
    description="Evaluate a Python expression safely",
    params={
        "expression": {"type": "string", "description": "Python expression to evaluate"},
    },
    required=["expression"],
)
async def sandbox_eval(expression: str) -> dict:
    """Evaluate a Python expression. Uses ast.literal_eval for literals, falls back to subprocess."""
    try:
        result = ast.literal_eval(expression)
        return {"ok": True, "result": str(result)}
    except (ValueError, SyntaxError):
        # Fall back to sandbox_exec — use safe wrapper to prevent code injection
        # Write expression to a variable, never interpolate into code string
        import json
        safe_expr = json.dumps(expression)
        code = (
            "import ast\n"
            f"_expr = {safe_expr}\n"
            "try:\n"
            "    _code = compile(_expr, '<sandbox_eval>', 'eval')\n"
            "    print(repr(eval(_code)))\n"
            "except Exception as e:\n"
            "    raise SystemExit(f'eval error: {e}')\n"
        )
        r = await sandbox_exec(code, timeout=10)
        if r["ok"]:
            return {"ok": True, "result": r["stdout"].strip()}
        return {"ok": False, "result": r["stderr"]}
