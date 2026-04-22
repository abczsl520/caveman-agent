"""Platform types — shared data structures for the gateway layer.

All platform adapters produce/consume these types. They are the contract
between the gateway runner and individual platform implementations.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Awaitable, Callable, Dict, List, Optional

__all__ = [
    "Platform",
    "MessageType",
    "ProcessingOutcome",
    "SessionSource",
    "MessageEvent",
    "SendResult",
    "PlatformConfig",
    "build_session_key",
]



class Platform(Enum):
    """Supported messaging platforms."""
    LOCAL = "local"
    DISCORD = "discord"
    TELEGRAM = "telegram"
    SLACK = "slack"
    WHATSAPP = "whatsapp"
    SIGNAL = "signal"
    MATRIX = "matrix"
    FEISHU = "feishu"
    WECOM = "wecom"
    WEIXIN = "weixin"
    DINGTALK = "dingtalk"
    EMAIL = "email"
    SMS = "sms"
    WEBHOOK = "webhook"
    API = "api"


class MessageType(Enum):
    """Types of incoming messages."""
    TEXT = "text"
    PHOTO = "photo"
    VIDEO = "video"
    AUDIO = "audio"
    VOICE = "voice"
    DOCUMENT = "document"
    STICKER = "sticker"
    LOCATION = "location"
    COMMAND = "command"


class ProcessingOutcome(Enum):
    """Result classification for message-processing lifecycle hooks."""
    SUCCESS = "success"
    FAILURE = "failure"
    CANCELLED = "cancelled"


@dataclass
class SessionSource:
    """Describes where a message originated from.

    Used to:
    1. Route responses back to the right place
    2. Inject context into the system prompt
    3. Build session keys for conversation tracking
    """
    platform: Platform
    chat_id: str
    chat_name: Optional[str] = None
    chat_type: str = "dm"  # "dm", "group", "channel", "thread"
    user_id: Optional[str] = None
    user_name: Optional[str] = None
    thread_id: Optional[str] = None
    chat_topic: Optional[str] = None

    @property
    def description(self) -> str:
        """Human-readable description of the source."""
        if self.platform == Platform.LOCAL:
            return "CLI terminal"
        parts = []
        if self.chat_type == "dm":
            parts.append(f"DM with {self.user_name or self.user_id or 'user'}")
        elif self.chat_type == "group":
            parts.append(f"group: {self.chat_name or self.chat_id}")
        elif self.chat_type == "channel":
            parts.append(f"channel: {self.chat_name or self.chat_id}")
        else:
            parts.append(self.chat_name or self.chat_id)
        if self.thread_id:
            parts.append(f"thread: {self.thread_id}")
        return ", ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "platform": self.platform.value,
            "chat_id": self.chat_id,
            "chat_name": self.chat_name,
            "chat_type": self.chat_type,
            "user_id": self.user_id,
            "user_name": self.user_name,
            "thread_id": self.thread_id,
            "chat_topic": self.chat_topic,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionSource":
        return cls(
            platform=Platform(data["platform"]),
            chat_id=str(data["chat_id"]),
            chat_name=data.get("chat_name"),
            chat_type=data.get("chat_type", "dm"),
            user_id=data.get("user_id"),
            user_name=data.get("user_name"),
            thread_id=data.get("thread_id"),
            chat_topic=data.get("chat_topic"),
        )


@dataclass
class MessageEvent:
    """Incoming message from a platform — normalized representation."""
    text: str
    message_type: MessageType = MessageType.TEXT
    source: Optional[SessionSource] = None
    raw_message: Any = None
    message_id: Optional[str] = None

    # Media attachments (local file paths after download)
    media_urls: List[str] = field(default_factory=list)
    media_types: List[str] = field(default_factory=list)

    # Reply context
    reply_to_message_id: Optional[str] = None
    reply_to_text: Optional[str] = None

    # Auto-loaded skill binding
    auto_skill: Optional[str] = None

    # Interaction flags
    is_mention: bool = False
    is_reply_to_bot: bool = False

    # Internal synthetic events bypass auth
    internal: bool = False

    timestamp: datetime = field(default_factory=datetime.now)

    def is_command(self) -> bool:
        return self.text.startswith("/")

    def get_command(self) -> Optional[str]:
        if not self.is_command():
            return None
        parts = self.text.split(maxsplit=1)
        raw = parts[0][1:].lower() if parts else None
        if raw and "@" in raw:
            raw = raw.split("@", 1)[0]
        if raw and "/" in raw:
            return None
        return raw

    def get_command_args(self) -> str:
        if not self.is_command():
            return self.text
        parts = self.text.split(maxsplit=1)
        return parts[1] if len(parts) > 1 else ""


@dataclass
class SendResult:
    """Result of sending a message."""
    success: bool
    message_id: Optional[str] = None
    error: Optional[str] = None
    raw_response: Any = None
    retryable: bool = False


@dataclass
class PlatformConfig:
    """Configuration for a single messaging platform."""
    enabled: bool = False
    token: Optional[str] = None
    api_key: Optional[str] = None
    home_chat_id: Optional[str] = None
    reply_to_mode: str = "first"  # "off", "first", "all"
    extra: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PlatformConfig":
        return cls(
            enabled=data.get("enabled", False),
            token=data.get("token"),
            api_key=data.get("api_key"),
            home_chat_id=data.get("home_chat_id"),
            reply_to_mode=data.get("reply_to_mode", "first"),
            extra={k: v for k, v in data.items()
                   if k not in ("enabled", "token", "api_key", "home_chat_id", "reply_to_mode")},
        )


# Type alias for message handlers
MessageHandler = Callable[[MessageEvent], Awaitable[Optional[str]]]


def build_session_key(
    source: SessionSource,
    *,
    group_sessions_per_user: bool = True,
    thread_sessions_per_user: bool = False,
) -> str:
    """Build a session key from a message source.

    Controls session isolation:
    - group_sessions_per_user=True: each user in a group gets their own session
    - thread_sessions_per_user=True: each user in a thread gets their own session
    """
    parts = [source.platform.value, source.chat_id]
    if source.thread_id:
        parts.append(f"t:{source.thread_id}")
        if thread_sessions_per_user and source.user_id:
            parts.append(f"u:{source.user_id}")
    elif source.chat_type in ("group", "channel"):
        if group_sessions_per_user and source.user_id:
            parts.append(f"u:{source.user_id}")
    return ":".join(parts)
