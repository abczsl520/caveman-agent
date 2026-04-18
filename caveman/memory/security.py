"""Memory content security scanner.

Ported from Hermes memory_tool.py (MIT, Nous Research).
Scans memory content for injection, exfiltration, and deception patterns
before storing. Memories are injected into system prompts, so they are
a high-value attack surface.
"""
from __future__ import annotations

import re
from typing import Optional

# Threat patterns: (regex, pattern_id)
_MEMORY_THREAT_PATTERNS = [
    # Prompt injection
    (r'ignore\s+(previous|all|above|prior)(\s+\w+)*\s+instructions', "prompt_injection"),
    (r'you\s+are\s+now\s+', "role_hijack"),
    (r'do\s+not\s+tell\s+the\s+user', "deception_hide"),
    (r'system\s+prompt\s+override', "sys_prompt_override"),
    (r'disregard\s+(your|all|any)\s+(instructions|rules|guidelines)', "disregard_rules"),
    (r'act\s+as\s+(if|though)\s+you\s+(have\s+no|don\'t\s+have)\s+'
     r'(restrictions|limits|rules)', "bypass_restrictions"),
    # Exfiltration
    (r'curl\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_curl"),
    (r'wget\s+[^\n]*\$\{?\w*(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL|API)', "exfil_wget"),
    (r'cat\s+[^\n]*(\.env|credentials|\.netrc|\.pgpass|\.npmrc|\.pypirc)', "read_secrets"),
    # Persistence / backdoor
    (r'authorized_keys', "ssh_backdoor"),
    (r'\$HOME/\.ssh|\~/\.ssh', "ssh_access"),
    (r'\$HOME/\.caveman/\.env|\~/\.caveman/\.env', "caveman_env"),
    # Data exfil via encoding
    (r'base64\s+[^\n]*(\.env|secret|key|token|password)', "exfil_base64"),
    (r'echo\s+[^\n]*>\s*[^\n]*/\.(bash|zsh|fish)rc', "rc_injection"),
]

# Invisible unicode characters used for injection
_INVISIBLE_CHARS = {
    '\u200b', '\u200c', '\u200d', '\u2060', '\ufeff',
    '\u202a', '\u202b', '\u202c', '\u202d', '\u202e',
}


def scan_memory_content(content: str) -> Optional[str]:
    """Scan memory content for threats. Returns error string if blocked, None if safe."""
    # Check invisible unicode
    for char in _INVISIBLE_CHARS:
        if char in content:
            return (
                f"Blocked: content contains invisible unicode character "
                f"U+{ord(char):04X} (possible injection)."
            )

    # Check threat patterns
    for pattern, pid in _MEMORY_THREAT_PATTERNS:
        if re.search(pattern, content, re.IGNORECASE):
            return (
                f"Blocked: content matches threat pattern '{pid}'. "
                f"Memory entries are injected into the system prompt and "
                f"must not contain injection or exfiltration payloads."
            )

    return None


def is_safe(content: str) -> bool:
    """Quick check: is this content safe to store?"""
    return scan_memory_content(content) is None
