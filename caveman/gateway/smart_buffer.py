"""SmartBuffer — Three-layer text buffer for gateway streaming.

Hermes-inspired architecture:
- Layer 1: _full_text — all text for conversation history
- Layer 2: interim flush — text before tool_calls sent as commentary
- Layer 3: final flush — last text block is the "real" reply
"""
from __future__ import annotations
import asyncio
import logging
import re

from caveman.gateway.router import GatewayRouter

logger = logging.getLogger("caveman.gateway")

class _SmartBuffer:
    """Three-layer text buffer (Hermes-inspired).

    Layer 1: _full_text — all text for conversation history
    Layer 2: interim flush — text before tool_calls sent as commentary
    Layer 3: final flush — last text block is the "real" reply

    Flush triggers (whichever comes first):
    1. Text reaches 1800 chars (Discord limit margin)
    2. Explicit flush() call (tool_call boundary, done)
    3. Silence timer: 5s without new text → auto-flush partial
    """

    CHAR_LIMIT = 1800
    SILENCE_TIMEOUT = 5.0

    # Patterns that indicate internal monologue (not user-facing)
    _MONOLOGUE_PATTERNS = re.compile(
        r'^(Now (?:let me|I)|Let me |OK,? (?:let me|I)|I\'ll |Alright,? |First,? let me |Next,? )'
        r'.{0,80}$',
        re.MULTILINE,
    )

    def __init__(self, router: GatewayRouter, gw: str, ch: str, interim_enabled: bool = True):
        self._router = router
        self._gw = gw
        self._ch = ch
        self._buf = ""
        self._full_text = ""       # All text added (for conversation history)
        self._sent_text = ""       # Text actually sent to user (interim + final)
        self._sent_count = 0
        self._interim_enabled = interim_enabled
        self._already_sent: set[str] = set()  # Dedup: content hashes of sent text
        self._timer: asyncio.TimerHandle | None = None

    async def add(self, text: str) -> None:
        self._buf += text
        self._full_text += text
        self._reset_timer()
        if len(self._buf) >= self.CHAR_LIMIT:
            await self.flush()

    def _reset_timer(self) -> None:
        if self._timer:
            self._timer.cancel()
        try:
            loop = asyncio.get_running_loop()
            self._timer = loop.call_later(
                self.SILENCE_TIMEOUT, lambda: asyncio.ensure_future(self.flush())
            )
        except RuntimeError:
            pass

    def _strip_think_blocks(self, text: str) -> str:
        """Remove <think>...</think> blocks (Hermes-style)."""
        return re.sub(r'<think>.*?</think>\s*', '', text, flags=re.DOTALL).strip()

    def _is_pure_monologue(self, text: str) -> bool:
        """Check if text is pure internal monologue (short English preamble).

        Only filters SHORT single-line English preambles.
        Multi-line or long text is always considered user-facing.
        """
        stripped = text.strip()
        if not stripped:
            return True
        # Multi-line or long text → not monologue
        if '\n' in stripped or len(stripped) > 120:
            return False
        # Check against monologue patterns
        return bool(self._MONOLOGUE_PATTERNS.match(stripped))

    def _content_hash(self, text: str) -> str:
        """Simple hash for dedup."""
        return text.strip()[:200]

    async def flush_interim(self) -> str:
        """Flush buffer as interim message (Layer 2).

        Called at tool_call boundary. Sends text to user if it's meaningful.
        Returns the flushed text (for logging).
        """
        if self._timer:
            self._timer.cancel()
            self._timer = None

        text = self._strip_think_blocks(self._buf).strip()
        self._buf = ""

        if not text:
            return ""

        # Check dedup
        h = self._content_hash(text)
        if h in self._already_sent:
            logger.info("Interim dedup: skipping already-sent text (%d chars)", len(text))
            return text

        # Check if pure monologue
        if self._is_pure_monologue(text):
            logger.info("Interim filtered monologue: %s", text[:80])
            return text

        # Send as interim message (if enabled)
        if self._interim_enabled:
            try:
                await self._router.send(self._gw, self._ch, text)
                self._sent_text += text + "\n"
                self._sent_count += 1
                self._already_sent.add(h)
                logger.info("Interim sent (%d chars): %s", len(text), text[:80])
            except Exception as e:
                logger.warning("Interim flush failed: %s", e)

        return text

    async def flush(self) -> None:
        """Flush buffer as final message (Layer 3)."""
        if self._timer:
            self._timer.cancel()
            self._timer = None
        text = self._strip_think_blocks(self._buf).strip()
        if not text:
            return
        self._buf = ""

        # Dedup: don't re-send if already sent as interim
        h = self._content_hash(text)
        if h in self._already_sent:
            logger.debug("Dedup: skipping already-sent final text")
            return

        try:
            await self._router.send(self._gw, self._ch, text)
            self._sent_text += text + "\n"
            self._sent_count += 1
            self._already_sent.add(h)
        except Exception as e:
            logger.warning("Buffer flush failed: %s", e)

    @property
    def sent_any(self) -> bool:
        return self._sent_count > 0

    def cancel(self) -> None:
        if self._timer:
            self._timer.cancel()
            self._timer = None


