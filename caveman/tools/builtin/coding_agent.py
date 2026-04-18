"""Coding agent tool — delegate coding tasks to external AI agents."""
from __future__ import annotations

from caveman.bridge.cli_agents import CLIAgentRunner
from caveman.tools.registry import tool


@tool(
    name="coding_agent",
    description="Delegate coding tasks to external AI agents (Claude Code, Codex)",
    params={
        "agent": {"type": "string", "description": "Agent name (claude, codex)", "default": "claude"},
        "task": {"type": "string", "description": "Task description / prompt"},
        "cwd": {"type": "string", "description": "Working directory", "default": "."},
        "timeout": {"type": "integer", "description": "Timeout in seconds", "default": 900},
    },
    required=["task"],
)
async def coding_agent_tool(
    task: str,
    agent: str = "claude",
    cwd: str = ".",
    timeout: int = 900,
) -> str:
    """Run an external coding agent and return its output."""
    runner = CLIAgentRunner()
    result = await runner.run(agent, task, cwd, timeout)
    return (
        f"Agent: {result.agent}\n"
        f"Exit: {result.exit_code}\n"
        f"Duration: {result.duration:.1f}s\n"
        f"Timed out: {result.timed_out}\n"
        f"Output:\n{result.output}"
    )
