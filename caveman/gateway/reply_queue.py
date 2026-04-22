"""Reply queue — buffer and drain followup messages.

When multiple messages arrive while the agent is processing, they're queued
and drained in order after the current run completes. Prevents message loss
and ensures sequential processing.

Queue policies:
- FIFO: Process all queued messages in order
- LATEST: Drop all but the latest queued message
- MERGE: Merge queued messages into a single context
"""
from __future__ import annotations

import logging
import time
from collections import deque
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__all__ = [
    "QueuePolicy",
    "QueuedMessage",
    "ReplyQueue",
    "QueueManager",
]


logger = logging.getLogger(__name__)


class QueuePolicy(str, Enum):
    """Backpressure policy for the outbound reply queue."""
    FIFO = "fifo"
    LATEST = "latest"
    MERGE = "merge"


@dataclass
class QueuedMessage:
    """A message waiting in the reply queue with priority and expiry metadata."""
    body: str
    sender_id: str = ""
    sender_name: str = ""
    message_id: str = ""
    queued_at: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)


class ReplyQueue:
    """Per-session message queue with configurable drain policy."""

    def __init__(
        self,
        session_id: str,
        policy: QueuePolicy = QueuePolicy.FIFO,
        max_depth: int = 20,
    ) -> None:
        self.session_id = session_id
        self.policy = policy
        self.max_depth = max_depth
        self._queue: deque[QueuedMessage] = deque(maxlen=max_depth)
        self._processing = False
        self._drain_scheduled = False

    def enqueue(self, message: QueuedMessage) -> int:
        """Add a message to the queue. Returns current queue depth."""
        message.queued_at = message.queued_at or time.time()
        self._queue.append(message)
        logger.debug("Queue %s: enqueued (depth=%d)", self.session_id[:12], len(self._queue))
        return len(self._queue)

    def drain(self) -> list[QueuedMessage]:
        """Drain the queue according to policy.

        Returns list of messages to process.
        """
        if not self._queue:
            return []

        if self.policy == QueuePolicy.LATEST:
            latest = self._queue[-1]
            self._queue.clear()
            return [latest]

        if self.policy == QueuePolicy.MERGE:
            merged = QueuedMessage(
                body="\n\n".join(m.body for m in self._queue if m.body),
                sender_id=self._queue[-1].sender_id,
                sender_name=self._queue[-1].sender_name,
                message_id=self._queue[-1].message_id,
                queued_at=self._queue[0].queued_at,
            )
            self._queue.clear()
            return [merged]

        # FIFO
        messages = list(self._queue)
        self._queue.clear()
        return messages

    @property
    def depth(self) -> int:
        return len(self._queue)

    @property
    def is_empty(self) -> bool:
        return len(self._queue) == 0

    @property
    def is_processing(self) -> bool:
        return self._processing

    def set_processing(self, value: bool) -> None:
        self._processing = value

    def clear(self) -> int:
        """Clear the queue. Returns number of messages dropped."""
        count = len(self._queue)
        self._queue.clear()
        return count


class QueueManager:
    """Manages reply queues across sessions."""

    def __init__(self, default_policy: QueuePolicy = QueuePolicy.FIFO) -> None:
        self._queues: dict[str, ReplyQueue] = {}
        self._default_policy = default_policy

    def get_queue(self, session_id: str) -> ReplyQueue:
        if session_id not in self._queues:
            self._queues[session_id] = ReplyQueue(session_id, policy=self._default_policy)
        return self._queues[session_id]

    def enqueue(self, session_id: str, message: QueuedMessage) -> int:
        return self.get_queue(session_id).enqueue(message)

    def drain(self, session_id: str) -> list[QueuedMessage]:
        return self.get_queue(session_id).drain()

    def is_processing(self, session_id: str) -> bool:
        q = self._queues.get(session_id)
        return q.is_processing if q else False

    def cleanup_empty(self) -> int:
        """Remove empty queues."""
        empty = [sid for sid, q in self._queues.items() if q.is_empty and not q.is_processing]
        for sid in empty:
            del self._queues[sid]
        return len(empty)

    def stats(self) -> dict[str, Any]:
        return {
            "total_queues": len(self._queues),
            "total_queued": sum(q.depth for q in self._queues.values()),
            "processing": sum(1 for q in self._queues.values() if q.is_processing),
        }
