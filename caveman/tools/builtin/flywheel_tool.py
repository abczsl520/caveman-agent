"""Flywheel tool — trigger self-audit from within agent conversations.

Allows Caveman to run its own flywheel (audit → fix → test → commit)
when asked via Discord or any other gateway.
"""
from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

from caveman.tools.registry import tool

logger = logging.getLogger(__name__)

PROJECT_DIR = Path(__file__).resolve().parents[3]  # caveman/tools/builtin -> caveman root


@tool(
    name="flywheel",
    description=(
        "Run Caveman's self-improvement flywheel: audit a subsystem, "
        "find bugs, fix them, run tests, commit. "
        "Use this when asked to run flywheel, self-audit, or improve Caveman's own code."
    ),
    params={
        "target": {
            "type": "string",
            "description": (
                "Subsystem to audit. One of: security, tools, memory, agent, "
                "compression, providers, gateway, config, wiki, coordinator, "
                "trajectory, skills, engines, bridge, mcp. "
                "If not specified, picks the next unaudited subsystem."
            ),
        },
        "rounds": {
            "type": "integer",
            "description": "Number of flywheel rounds to run (default 1)",
        },
        "stats_only": {
            "type": "boolean",
            "description": "If true, just return flywheel statistics without running",
        },
    },
    required=[],
)
async def flywheel_exec(
    target: str | None = None,
    rounds: int = 1,
    stats_only: bool = False,
) -> dict:
    """Execute flywheel audit/fix cycle."""
    if stats_only:
        from caveman.cli.flywheel import FlywheelStats
        stats = FlywheelStats().summary()
        return {"ok": True, "stats": stats}

    # Run flywheel in subprocess to avoid blocking the gateway event loop
    cmd = [sys.executable, "-m", "caveman", "flywheel"]
    if target:
        cmd.extend(["--target", target])
    cmd.extend(["--rounds", str(rounds)])

    logger.info("Flywheel starting: target=%s rounds=%d", target or "auto", rounds)

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            cwd=str(PROJECT_DIR),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.STDOUT,
            env={**__import__("os").environ, "PYTHONUNBUFFERED": "1"},
        )

        output_lines = []
        while True:
            line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=600.0
            )
            if not line:
                break
            decoded = line.decode("utf-8", errors="replace").rstrip()
            output_lines.append(decoded)
            logger.info("flywheel: %s", decoded)

        await proc.wait()
        output = "\n".join(output_lines[-100:])  # Keep last 100 lines

        if proc.returncode == 0:
            return {
                "ok": True,
                "message": f"Flywheel completed ({target or 'auto'}, {rounds} round(s))",
                "output": output,
            }
        else:
            return {
                "ok": False,
                "error": f"Flywheel exited with code {proc.returncode}",
                "output": output,
            }

    except asyncio.TimeoutError:
        return {"ok": False, "error": "Flywheel timed out after 10 minutes"}
    except Exception as e:
        logger.exception("Flywheel failed: %s", e)
        return {"ok": False, "error": str(e)}
