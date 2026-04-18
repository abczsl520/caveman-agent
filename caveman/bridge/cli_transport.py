"""CLI transport — Layer 1 OpenClaw bridge via subprocess.

Pros: handles all auth, always works, no WebSocket complexity
Cons: per-command overhead (~200ms), no streaming, no events
"""
from __future__ import annotations

import asyncio
import json
import logging
import shutil
from typing import Any

logger = logging.getLogger(__name__)


class CLITransport:
    """Transport via `openclaw` CLI subprocess."""

    def __init__(self, session_key: str, agent_id: str):
        self._session_key = session_key
        self._agent_id = agent_id
        self._openclaw_path = shutil.which("openclaw") or "openclaw"
        self._tools_cache: list[dict] = []

    @property
    def name(self) -> str:
        return "cli"

    async def connect(self) -> bool:
        try:
            result = await self._run_cli(["health", "--json"])
            if result["returncode"] == 0:
                logger.info("OpenClaw CLI transport connected (gateway healthy)")
                return True
            logger.warning("OpenClaw gateway not healthy: %s", result["stderr"])
            return False
        except Exception as e:
            logger.error("Failed to connect via CLI: %s", e)
            return False

    async def disconnect(self) -> None:
        pass

    async def send_message(self, message: str, channel: str | None, target: str | None) -> dict:
        args = ["message", "send", "--message", message, "--json"]
        if channel:
            args.extend(["--channel", channel])
        if target:
            args.extend(["--target", target])
        result = await self._run_cli(args)
        return self._parse_result(result)

    async def agent_turn(self, message: str, **kwargs) -> dict:
        args = ["agent", "--message", message, "--json",
                "--session-id", self._session_key]
        channel = kwargs.get("channel")
        if channel:
            args.extend(["--channel", channel])
        target = kwargs.get("target")
        if target:
            args.extend(["--to", target])
        if kwargs.get("deliver", False):
            args.append("--deliver")
        timeout = kwargs.get("timeout", 120)
        args.extend(["--timeout", str(timeout)])
        result = await self._run_cli(args, timeout=timeout + 10)
        return self._parse_result(result)

    async def list_tools(self) -> list[dict]:
        self._tools_cache = KNOWN_MCP_TOOLS
        return self._tools_cache

    async def call_tool(self, tool_name: str, args: dict) -> dict:
        if tool_name in ("messages_send", "message"):
            return await self.send_message(
                message=args.get("message", ""),
                channel=args.get("channel"),
                target=args.get("target") or args.get("to"),
            )
        elif tool_name == "conversations_list":
            result = await self._run_cli(["directory", "groups", "--json"])
            return self._parse_result(result)
        elif tool_name == "sessions_spawn":
            spawn_args = ["acp", "client"]
            if args.get("agentId"):
                spawn_args.extend(["--agent", args["agentId"]])
            if args.get("task"):
                spawn_args.append(args["task"])
            result = await self._run_cli(spawn_args, timeout=args.get("runTimeoutSeconds", 300))
            return self._parse_result(result)
        elif tool_name == "sessions_send":
            send_args = ["sessions", "send", args.get("sessionKey", ""), args.get("message", "")]
            result = await self._run_cli(send_args)
            return self._parse_result(result)
        else:
            return {"error": f"Tool '{tool_name}' not mapped to CLI transport", "success": False}

    def get_tool_schemas(self) -> list[dict]:
        return [
            {
                "name": f"openclaw_{t['name']}",
                "description": f"[OpenClaw] {t.get('description', '')}",
                "input_schema": t.get("inputSchema", {"type": "object"}),
            }
            for t in self._tools_cache
        ]

    async def _run_cli(self, args: list[str], timeout: int = 30) -> dict:
        cmd = [self._openclaw_path] + args
        logger.debug("CLI: %s", " ".join(cmd))
        try:
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return {
                "stdout": stdout.decode("utf-8", errors="replace").strip(),
                "stderr": stderr.decode("utf-8", errors="replace").strip(),
                "returncode": proc.returncode or 0,
            }
        except asyncio.TimeoutError:
            proc.kill()
            return {"stdout": "", "stderr": "Command timed out", "returncode": -1}
        except Exception as e:
            return {"stdout": "", "stderr": str(e), "returncode": -1}

    @staticmethod
    def _parse_result(result: dict) -> dict:
        if result["returncode"] != 0:
            return {"error": result["stderr"], "success": False}
        try:
            return json.loads(result["stdout"])
        except json.JSONDecodeError:
            return {"result": result["stdout"], "success": True}


# Known tools from OpenClaw
KNOWN_MCP_TOOLS = [
    {
        "name": "messages_send",
        "description": "Send a message to a conversation",
        "inputSchema": {
            "type": "object",
            "properties": {
                "message": {"type": "string"},
                "channel": {"type": "string"},
                "target": {"type": "string"},
            },
            "required": ["message"],
        },
    },
    {
        "name": "conversations_list",
        "description": "List available conversations",
        "inputSchema": {"type": "object", "properties": {}},
    },
    {
        "name": "sessions_spawn",
        "description": "Spawn an ACP agent session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "task": {"type": "string"},
                "agentId": {"type": "string"},
                "runtime": {"type": "string"},
                "mode": {"type": "string"},
            },
            "required": ["task"],
        },
    },
    {
        "name": "sessions_send",
        "description": "Send a message to an agent session",
        "inputSchema": {
            "type": "object",
            "properties": {
                "sessionKey": {"type": "string"},
                "message": {"type": "string"},
            },
            "required": ["sessionKey", "message"],
        },
    },
]
