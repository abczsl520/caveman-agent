"""OpenAI-compatible shim that forwards requests to `copilot --acp`.

Full JSONRPC protocol over stdio with subprocess lifecycle management.
"""
from __future__ import annotations

import json
import logging
import os
import queue
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List, Optional, Tuple

from caveman.acp.copilot_protocol import (
    _resolve_command, _resolve_args, _jsonrpc_error,
    _ensure_path_within_cwd, _format_messages_as_prompt,
    _extract_tool_calls_from_text, ACP_MARKER_BASE_URL,
    _DEFAULT_TIMEOUT_SECONDS,
)

__all__ = [
    "CopilotMessage",
    "CopilotSession",
    "CopilotACPClient",
]


logger = logging.getLogger(__name__)


# ── OpenAI-compatible facade ───────────────────────────────────────────────

@dataclass
class CopilotMessage:
    """A message in a copilot session."""
    role: str
    content: str
    tool_calls: List[Any] = field(default_factory=list)
    reasoning: Optional[str] = None


@dataclass
class CopilotSession:
    """Tracks a copilot ACP session."""
    session_id: str
    cwd: str
    messages: List[CopilotMessage] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    status: str = "active"


class _ACPChatCompletions:
    def __init__(self, client: "CopilotACPClient"):
        self._client = client

    def create(self, **kwargs: Any) -> Any:
        return self._client._create_chat_completion(**kwargs)


class _ACPChatNamespace:
    def __init__(self, client: "CopilotACPClient"):
        self.completions = _ACPChatCompletions(client)


class CopilotACPClient:
    """Minimal OpenAI-client-compatible facade for Copilot ACP.

    Starts a short-lived ACP session per request, sends formatted
    conversation as a single prompt, collects text chunks, and converts
    back into the shape Caveman expects from an OpenAI client.
    """

    def __init__(
        self,
        *,
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        acp_command: Optional[str] = None,
        acp_args: Optional[List[str]] = None,
        acp_cwd: Optional[str] = None,
        **_: Any,
    ):
        self.api_key = api_key or "copilot-acp"
        self.base_url = base_url or ACP_MARKER_BASE_URL
        self._acp_command = acp_command or _resolve_command()
        self._acp_args = list(acp_args or _resolve_args())
        self._acp_cwd = str(Path(acp_cwd or os.getcwd()).resolve())
        self.chat = _ACPChatNamespace(self)
        self.is_closed = False
        self._active_process: Optional[subprocess.Popen] = None
        self._active_process_lock = threading.Lock()
        self._sessions: Dict[str, CopilotSession] = {}

    def close(self) -> None:
        """Terminate the active ACP process."""
        proc: Optional[subprocess.Popen]
        with self._active_process_lock:
            proc = self._active_process
            self._active_process = None
        self.is_closed = True
        if proc is None:
            return
        try:
            proc.terminate()
            proc.wait(timeout=2)
        except Exception:
            try:
                proc.kill()
            except Exception as exc:
                logger.debug("close: suppressed %s", exc)

    def create_session(self, session_id: str = "", **metadata) -> CopilotSession:
        """Create a tracked session."""
        import uuid
        sid = session_id or uuid.uuid4().hex[:12]
        session = CopilotSession(session_id=sid, cwd=self._acp_cwd)
        self._sessions[sid] = session
        return session

    def get_session(self, session_id: str) -> Optional[CopilotSession]:
        return self._sessions.get(session_id)

    def list_sessions(self) -> List[Dict[str, Any]]:
        return [
            {"session_id": s.session_id, "status": s.status, "messages": len(s.messages)}
            for s in self._sessions.values()
        ]

    def close_session(self, session_id: str) -> bool:
        session = self._sessions.get(session_id)
        if session:
            session.status = "closed"
            return True
        return False

    def _create_chat_completion(
        self,
        *,
        model: Optional[str] = None,
        messages: Optional[List[Dict[str, Any]]] = None,
        timeout: Optional[float] = None,
        tools: Optional[List[Dict[str, Any]]] = None,
        tool_choice: Any = None,
        **_: Any,
    ) -> Any:
        """Create a chat completion via ACP subprocess."""
        prompt_text = _format_messages_as_prompt(
            messages or [], model=model, tools=tools, tool_choice=tool_choice,
        )
        response_text, reasoning_text = self._run_prompt(
            prompt_text, timeout_seconds=float(timeout or _DEFAULT_TIMEOUT_SECONDS),
        )

        tool_calls, cleaned_text = _extract_tool_calls_from_text(response_text)

        usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
            prompt_tokens_details=SimpleNamespace(cached_tokens=0),
        )
        assistant_message = SimpleNamespace(
            content=cleaned_text, tool_calls=tool_calls,
            reasoning=reasoning_text or None,
            reasoning_content=reasoning_text or None,
            reasoning_details=None,
        )
        finish_reason = "tool_calls" if tool_calls else "stop"
        choice = SimpleNamespace(message=assistant_message, finish_reason=finish_reason)
        return SimpleNamespace(choices=[choice], usage=usage, model=model or "copilot-acp")

    def _run_prompt(self, prompt_text: str, *, timeout_seconds: float) -> Tuple[str, str]:
        """Start ACP subprocess, send prompt, collect response."""
        try:
            proc = subprocess.Popen(
                [self._acp_command] + self._acp_args,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1, cwd=self._acp_cwd,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Could not start Copilot ACP command '{self._acp_command}'. "
                "Install GitHub Copilot CLI or set CAVEMAN_COPILOT_ACP_COMMAND."
            ) from exc

        if proc.stdin is None or proc.stdout is None:
            proc.kill()
            raise RuntimeError("ACP process did not expose stdin/stdout pipes.")

        self.is_closed = False
        with self._active_process_lock:
            self._active_process = proc

        inbox: queue.Queue[Dict[str, Any]] = queue.Queue()
        stderr_tail: deque = deque(maxlen=40)

        def _stdout_reader() -> None:
            assert proc.stdout is not None
            for line in proc.stdout:
                try:
                    inbox.put(json.loads(line))
                except Exception:
                    inbox.put({"raw": line.rstrip("\n")})

        def _stderr_reader() -> None:
            if proc.stderr is None:
                return
            for line in proc.stderr:
                stderr_tail.append(line.rstrip("\n"))

        out_thread = threading.Thread(target=_stdout_reader, daemon=True)
        err_thread = threading.Thread(target=_stderr_reader, daemon=True)
        out_thread.start()
        err_thread.start()

        next_id = 0

        def _request(method: str, params: Dict[str, Any], *,
                     text_parts: Optional[List[str]] = None,
                     reasoning_parts: Optional[List[str]] = None) -> Any:
            nonlocal next_id
            next_id += 1
            request_id = next_id
            payload = {"jsonrpc": "2.0", "id": request_id, "method": method, "params": params}
            assert proc.stdin is not None
            proc.stdin.write(json.dumps(payload) + "\n")
            proc.stdin.flush()

            deadline = time.time() + timeout_seconds
            while time.time() < deadline:
                if proc.poll() is not None:
                    break
                try:
                    msg = inbox.get(timeout=0.1)
                except queue.Empty:
                    continue

                if self._handle_server_message(
                    msg, process=proc, cwd=self._acp_cwd,
                    text_parts=text_parts, reasoning_parts=reasoning_parts,
                ):
                    continue

                if msg.get("id") != request_id:
                    continue
                if "error" in msg:
                    err = msg.get("error") or {}
                    raise RuntimeError(f"ACP {method} failed: {err.get('message') or err}")
                return msg.get("result")

            stderr_text = "\n".join(stderr_tail).strip()
            if proc.poll() is not None and stderr_text:
                raise RuntimeError(f"ACP process exited early: {stderr_text}")
            raise TimeoutError(f"Timed out waiting for ACP response to {method}.")

        try:
            _request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": True, "writeTextFile": True}},
                "clientInfo": {"name": "caveman-agent", "title": "Caveman Agent", "version": "0.1.0"},
            })
            session = _request("session/new", {"cwd": self._acp_cwd, "mcpServers": []}) or {}
            session_id = str(session.get("sessionId") or "").strip()
            if not session_id:
                raise RuntimeError("ACP did not return a sessionId.")

            text_parts: List[str] = []
            reasoning_parts: List[str] = []
            _request(
                "session/prompt",
                {"sessionId": session_id, "prompt": [{"type": "text", "text": prompt_text}]},
                text_parts=text_parts, reasoning_parts=reasoning_parts,
            )
            return "".join(text_parts), "".join(reasoning_parts)
        finally:
            self.close()

    def _handle_server_message(
        self, msg: Dict[str, Any], *, process: subprocess.Popen,
        cwd: str, text_parts: Optional[List[str]], reasoning_parts: Optional[List[str]],
    ) -> bool:
        """Handle server-initiated messages (updates, fs requests, permissions)."""
        method = msg.get("method")
        if not isinstance(method, str):
            return False

        if method == "session/update":
            params = msg.get("params") or {}
            update = params.get("update") or {}
            kind = str(update.get("sessionUpdate") or "").strip()
            content = update.get("content") or {}
            chunk_text = ""
            if isinstance(content, dict):
                chunk_text = str(content.get("text") or "")
            if kind == "agent_message_chunk" and chunk_text and text_parts is not None:
                text_parts.append(chunk_text)
            elif kind == "agent_thought_chunk" and chunk_text and reasoning_parts is not None:
                reasoning_parts.append(chunk_text)
            return True

        if process.stdin is None:
            return True

        message_id = msg.get("id")
        params = msg.get("params") or {}

        if method == "session/request_permission":
            response = {
                "jsonrpc": "2.0", "id": message_id,
                "result": {"outcome": {"outcome": "allow_once"}},
            }
        elif method == "fs/read_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                content = path.read_text(encoding="utf-8") if path.exists() else ""
                line = params.get("line")
                limit = params.get("limit")
                if isinstance(line, int) and line > 1:
                    lines = content.splitlines(keepends=True)
                    start = line - 1
                    end = start + limit if isinstance(limit, int) and limit > 0 else None
                    content = "".join(lines[start:end])
                response = {"jsonrpc": "2.0", "id": message_id, "result": {"content": content}}
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        elif method == "fs/write_text_file":
            try:
                path = _ensure_path_within_cwd(str(params.get("path") or ""), cwd)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(str(params.get("content") or ""), encoding="utf-8")
                response = {"jsonrpc": "2.0", "id": message_id, "result": None}
            except Exception as exc:
                response = _jsonrpc_error(message_id, -32602, str(exc))
        else:
            response = _jsonrpc_error(
                message_id, -32601,
                f"ACP client method '{method}' is not supported yet.",
            )

        process.stdin.write(json.dumps(response) + "\n")
        process.stdin.flush()
        return True
