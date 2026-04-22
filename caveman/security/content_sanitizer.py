"""External content sanitization — clean untrusted input.

Prevents prompt injection and other attacks from external content
(web pages, user uploads, API responses, etc.).
"""
from __future__ import annotations

import re

__all__ = ["sanitize_external_content", "detect_prompt_injection"]

# Patterns that look like prompt injection attempts
_INJECTION_PATTERNS: list[tuple[str, re.Pattern]] = [
    ("role_override", re.compile(
        r"(you are now|ignore.*previous.*instructions|forget.*rules|"
        r"system prompt|override.*instructions|new instructions)",
        re.IGNORECASE,
    )),
    ("hidden_instruction", re.compile(
        r"(\[INST\]|\[/INST\]|<\|im_start\|>|<\|im_end\|>|"
        r"<\|system\|>|<\|user\|>|<\|assistant\|>)",
    )),
    ("data_exfil_request", re.compile(
        r"(send.*to.*http|post.*to.*url|exfiltrate|leak.*data|"
        r"upload.*secret|share.*api.?key)",
        re.IGNORECASE,
    )),
]

# Maximum length for external content (prevent context flooding)
MAX_EXTERNAL_CONTENT_LENGTH = 50_000


def detect_prompt_injection(text: str) -> list[tuple[str, str]]:
    """Detect potential prompt injection patterns.

    Returns list of (pattern_name, matched_text) tuples.
    """
    findings = []
    for name, pattern in _INJECTION_PATTERNS:
        matches = pattern.findall(text)
        for match in matches:
            matched = match if isinstance(match, str) else match[0]
            findings.append((name, matched[:100]))
    return findings


def sanitize_external_content(
    text: str,
    source: str = "unknown",
    max_length: int = MAX_EXTERNAL_CONTENT_LENGTH,
    strip_injection: bool = True,
) -> str:
    """Sanitize external content for safe inclusion in prompts.

    - Truncates to max_length
    - Optionally strips detected injection patterns
    - Wraps in clear boundary markers
    """
    if not text:
        return ""

    # Truncate
    if len(text) > max_length:
        text = text[:max_length] + f"\n\n[... truncated, {len(text) - max_length:,} chars omitted]"

    # Strip injection patterns
    if strip_injection:
        for name, pattern in _INJECTION_PATTERNS:
            text = pattern.sub(f"[{name} removed]", text)

    # Wrap in boundary markers
    return f"[External content from {source}]\n{text}\n[End external content]"
