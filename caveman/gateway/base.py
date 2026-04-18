"""Abstract Gateway base class."""
from __future__ import annotations
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from typing import Any, Callable


class Gateway(ABC):
    """Abstract base for messaging gateways (Discord, Telegram, etc.)."""

    _running: bool = False

    @abstractmethod
    async def start(self) -> None: ...

    @abstractmethod
    async def stop(self) -> None: ...

    @abstractmethod
    async def send_message(self, channel_id: str, text: str) -> None: ...

    @abstractmethod
    async def on_message(self, handler: Callable) -> None: ...

    @property
    @abstractmethod
    def name(self) -> str: ...

    @property
    def is_running(self) -> bool:
        return self._running

    async def send_streaming(self, channel_id: str, stream: AsyncIterator[Any]) -> dict:
        """Send streaming response. Default: collect all tokens, send as one.

        Subclasses (Discord/Telegram) can override to edit messages in real-time.
        """
        from caveman.agent.stream import StreamEvent
        parts: list[str] = []
        async for event in stream:
            if isinstance(event, StreamEvent) and event.type == "token":
                parts.append(str(event.data))
        text = "".join(parts)
        if text:
            await self.send_message(channel_id, text)
        return {"ok": True, "length": len(text)}
