"""Command Approval — dangerous command detection and approval flow.

Detects dangerous shell commands and manages approval state
per session. Extracted from Hermes tools/approval.py (923 lines).
"""
from __future__ import annotations

import contextvars
import hashlib
import logging
import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, Set

__all__ = [
    "set_current_session_key",
    "get_current_session_key",
    "DangerousCommandResult",
    "detect_dangerous_command",
    "ApprovalEntry",
    "ApprovalManager",
]


logger = logging.getLogger("caveman.tools.approval")

# ── Session Context ──

_current_session_key: contextvars.ContextVar[str] = contextvars.ContextVar(
    "approval_session_key", default="default",
)


def set_current_session_key(key: str) -> contextvars.Token:
    return _current_session_key.set(key)


def get_current_session_key() -> str:
    return _current_session_key.get("default")


# ── Dangerous Command Detection ──

_DANGEROUS_PATTERNS = [
    (re.compile(r"\brm\s+(-[rfRF]+\s+)?/"), "rm_root", "Removing files from root"),
    (re.compile(r"\brm\s+-[rfRF]*\s"), "rm_recursive", "Recursive file deletion"),
    (re.compile(r"\bmkfs\b"), "mkfs", "Filesystem formatting"),
    (re.compile(r"\bdd\s+"), "dd", "Direct disk write"),
    (re.compile(r"\b(shutdown|reboot|halt|poweroff)\b"), "power", "System power control"),
    (re.compile(r"\bchmod\s+777\b"), "chmod_777", "World-writable permissions"),
    (re.compile(r"\bchown\s+-R\b"), "chown_recursive", "Recursive ownership change"),
    (re.compile(r"\bcurl\s+.*\|\s*(bash|sh|zsh)\b"), "pipe_to_shell", "Piping URL to shell"),
    (re.compile(r"\bwget\s+.*\|\s*(bash|sh|zsh)\b"), "pipe_to_shell", "Piping URL to shell"),
    (re.compile(r"\bgit\s+push\s+.*--force\b"), "force_push", "Force push to git"),
    (re.compile(r"\bgit\s+reset\s+--hard\b"), "git_reset_hard", "Hard git reset"),
    (re.compile(r"\bgit\s+clean\s+-fd\b"), "git_clean", "Git clean force"),
    (re.compile(r"\bdrop\s+database\b", re.IGNORECASE), "drop_db", "Database drop"),
    (re.compile(r"\bdrop\s+table\b", re.IGNORECASE), "drop_table", "Table drop"),
    (re.compile(r"\btruncate\s+table\b", re.IGNORECASE), "truncate_table", "Table truncate"),
    (re.compile(r"\bkill\s+-9\b"), "kill_9", "Force kill process"),
    (re.compile(r"\bsudo\s+"), "sudo", "Elevated privileges"),
    (re.compile(r"\bnpm\s+publish\b"), "npm_publish", "Package publish"),
    (re.compile(r"\bpip\s+install\s+(?!-r\b)(?!--requirement)"), "pip_install", "Package installation"),
    (re.compile(r"\bdocker\s+rm\b"), "docker_rm", "Docker container removal"),
    (re.compile(r"\bdocker\s+system\s+prune\b"), "docker_prune", "Docker system prune"),
]


@dataclass
class DangerousCommandResult:
    """Result of dangerous command detection."""
    is_dangerous: bool = False
    pattern_key: str = ""
    description: str = ""
    command: str = ""
    approval_hash: str = ""


def detect_dangerous_command(command: str) -> DangerousCommandResult:
    """Detect if a command is dangerous."""
    normalized = _normalize_command(command)

    for pattern, key, description in _DANGEROUS_PATTERNS:
        if pattern.search(normalized):
            approval_hash = hashlib.sha256(
                f"{key}:{command}".encode()
            ).hexdigest()[:12]
            return DangerousCommandResult(
                is_dangerous=True,
                pattern_key=key,
                description=description,
                command=command,
                approval_hash=approval_hash,
            )

    return DangerousCommandResult(command=command)


def _normalize_command(command: str) -> str:
    """Normalize a command for detection."""
    # Remove leading whitespace and common prefixes
    command = command.strip()
    # Remove env var assignments at the start
    command = re.sub(r"^(\w+=\S+\s+)+", "", command)
    return command


# ── Approval Store ──

@dataclass
class ApprovalEntry:
    """An approval entry."""
    pattern_key: str
    policy: str = "allow-once"  # allow-once | allow-always | deny
    approved_at: float = 0
    approved_by: str = ""
    command_hash: str = ""


class ApprovalManager:
    """Manages command approvals per session."""

    def __init__(self):
        self._session_approvals: Dict[str, Dict[str, ApprovalEntry]] = {}
        self._permanent_approvals: Dict[str, ApprovalEntry] = {}
        self._yolo_sessions: Set[str] = set()
        self._pending: Dict[str, Dict[str, Any]] = {}
        self._notify_callbacks: Dict[str, Callable] = {}

    def is_approved(self, session_key: str, pattern_key: str) -> bool:
        """Check if a pattern is approved for a session."""
        if session_key in self._yolo_sessions:
            return True
        if pattern_key in self._permanent_approvals:
            return True
        session = self._session_approvals.get(session_key, {})
        entry = session.get(pattern_key)
        if not entry:
            return False
        if entry.policy == "allow-always":
            return True
        if entry.policy == "allow-once":
            # Consume the approval
            del session[pattern_key]
            return True
        return False

    def approve(
        self,
        session_key: str,
        pattern_key: str,
        policy: str = "allow-once",
    ) -> None:
        """Approve a pattern for a session."""
        if session_key not in self._session_approvals:
            self._session_approvals[session_key] = {}
        self._session_approvals[session_key][pattern_key] = ApprovalEntry(
            pattern_key=pattern_key,
            policy=policy,
            approved_at=time.time(),
        )

    def approve_permanent(self, pattern_key: str) -> None:
        """Permanently approve a pattern across all sessions."""
        self._permanent_approvals[pattern_key] = ApprovalEntry(
            pattern_key=pattern_key,
            policy="allow-always",
            approved_at=time.time(),
        )

    def enable_yolo(self, session_key: str) -> None:
        """Enable YOLO mode (auto-approve everything) for a session."""
        self._yolo_sessions.add(session_key)

    def disable_yolo(self, session_key: str) -> None:
        self._yolo_sessions.discard(session_key)

    def is_yolo(self, session_key: str) -> bool:
        return session_key in self._yolo_sessions

    def submit_pending(self, session_key: str, approval: Dict[str, Any]) -> None:
        """Submit a pending approval request."""
        self._pending[session_key] = approval
        # Notify gateway
        cb = self._notify_callbacks.get(session_key)
        if cb:
            try:
                cb(approval)
            except Exception as exc:
                logger.debug("submit_pending: suppressed %s", exc)

    def has_pending(self, session_key: str) -> bool:
        return session_key in self._pending

    def resolve_pending(self, session_key: str, choice: str) -> bool:
        """Resolve a pending approval."""
        if session_key not in self._pending:
            return False
        pending = self._pending.pop(session_key)
        pattern_key = pending.get("pattern_key", "")
        if choice in ("allow-once", "allow-always"):
            self.approve(session_key, pattern_key, choice)
            return True
        return False

    def register_notify(self, session_key: str, callback: Callable) -> None:
        self._notify_callbacks[session_key] = callback

    def unregister_notify(self, session_key: str) -> None:
        self._notify_callbacks.pop(session_key, None)

    def check_and_request(
        self,
        session_key: str,
        command: str,
    ) -> Dict[str, Any]:
        """Check a command and request approval if needed."""
        result = detect_dangerous_command(command)
        if not result.is_dangerous:
            return {"approved": True, "command": command}

        if self.is_approved(session_key, result.pattern_key):
            return {"approved": True, "command": command, "pre_approved": True}

        # Submit pending approval
        self.submit_pending(session_key, {
            "pattern_key": result.pattern_key,
            "description": result.description,
            "command": command,
            "approval_hash": result.approval_hash,
        })

        return {
            "approved": False,
            "pending": True,
            "pattern_key": result.pattern_key,
            "description": result.description,
            "approval_hash": result.approval_hash,
            "message": (
                f"⚠️ Dangerous command detected: {result.description}\n"
                f"Command: `{command}`\n"
                f"Approve with: /approve {result.approval_hash} allow-once"
            ),
        }
