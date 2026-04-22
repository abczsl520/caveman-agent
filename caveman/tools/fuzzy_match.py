"""Fuzzy matching for file edit operations.

8-strategy matching chain to robustly find and replace text,
accommodating LLM-generated code variations in whitespace, indentation, escaping.

Strategies (tried in order):
1. Exact match
2. Line-trimmed (strip leading/trailing whitespace per line)
3. Whitespace normalized (collapse multiple spaces)
4. Indentation flexible (ignore indentation)
5. Escape normalized (\\n literals → actual newlines)
6. Trimmed boundary (trim first/last line only)
7. Block anchor (match first+last lines, similarity for middle)
8. Context-aware (50% line similarity threshold)
"""
from __future__ import annotations

import re
from difflib import SequenceMatcher

_UNICODE_MAP = {
    "\u201c": '"', "\u201d": '"',
    "\u2018": "'", "\u2019": "'",
    "\u2014": "--", "\u2013": "-",
    "\u2026": "...", "\u00a0": " ",
}


def fuzzy_find_and_replace(
    content: str,
    old_string: str,
    new_string: str,
    replace_all: bool = False,
) -> tuple[str, int, str, str | None]:
    """Find and replace text using fuzzy matching.

    Returns (new_content, match_count, strategy_used, error_message).
    """
    if not old_string:
        return content, 0, "", "empty search string"

    # Normalize unicode in search string
    normalized_old = _normalize_unicode(old_string)
    normalized_content = _normalize_unicode(content)

    strategies = [
        ("exact", _strategy_exact),
        ("line_trimmed", _strategy_line_trimmed),
        ("whitespace_normalized", _strategy_whitespace_normalized),
        ("indentation_flexible", _strategy_indentation_flexible),
        ("escape_normalized", _strategy_escape_normalized),
        ("trimmed_boundary", _strategy_trimmed_boundary),
        ("block_anchor", _strategy_block_anchor),
        ("context_aware", _strategy_context_aware),
    ]

    for name, strategy in strategies:
        result = strategy(content, normalized_content, old_string, normalized_old, new_string, replace_all)
        if result is not None:
            new_content, count = result
            return new_content, count, name, None

    return content, 0, "", f"no match found for:\n{old_string[:200]}"


def _normalize_unicode(text: str) -> str:
    for old, new in _UNICODE_MAP.items():
        text = text.replace(old, new)
    return text


def _strategy_exact(content, norm_content, old, norm_old, new, replace_all):
    if norm_old in norm_content:
        count = norm_content.count(norm_old) if replace_all else 1
        # Use original content for replacement
        if old in content:
            result = content.replace(old, new) if replace_all else content.replace(old, new, 1)
        else:
            result = norm_content.replace(norm_old, new) if replace_all else norm_content.replace(norm_old, new, 1)
        return result, count
    return None


def _strategy_line_trimmed(content, norm_content, old, norm_old, new, replace_all):
    trimmed_old = "\n".join(line.strip() for line in norm_old.splitlines())
    trimmed_content = "\n".join(line.strip() for line in norm_content.splitlines())
    if trimmed_old in trimmed_content:
        # Find the actual lines in content that match
        return _replace_by_trimmed_lines(content, old, new, replace_all)
    return None


def _strategy_whitespace_normalized(content, norm_content, old, norm_old, new, replace_all):
    ws_old = re.sub(r'[ \t]+', ' ', norm_old)
    ws_content = re.sub(r'[ \t]+', ' ', norm_content)
    if ws_old in ws_content:
        # For single-line, do direct replacement in ws-normalized space
        if "\n" not in ws_old:
            # Find the actual substring in content that matches when ws-normalized
            for i in range(len(content)):
                for j in range(i + 1, len(content) + 1):
                    candidate = content[i:j]
                    if re.sub(r'[ \t]+', ' ', candidate) == ws_old:
                        result = content[:i] + new + content[j:]
                        return result, 1
        return _replace_by_trimmed_lines(content, old, new, replace_all)
    return None


def _strategy_indentation_flexible(content, norm_content, old, norm_old, new, replace_all):
    stripped_old = "\n".join(line.lstrip() for line in norm_old.splitlines())
    stripped_content = "\n".join(line.lstrip() for line in norm_content.splitlines())
    if stripped_old in stripped_content:
        return _replace_by_trimmed_lines(content, old, new, replace_all)
    return None


def _strategy_escape_normalized(content, norm_content, old, norm_old, new, replace_all):
    esc_old = norm_old.replace("\\n", "\n").replace("\\t", "\t")
    if esc_old in norm_content:
        result = norm_content.replace(esc_old, new) if replace_all else norm_content.replace(esc_old, new, 1)
        return result, norm_content.count(esc_old) if replace_all else 1
    return None


def _strategy_trimmed_boundary(content, norm_content, old, norm_old, new, replace_all):
    old_lines = norm_old.splitlines()
    if len(old_lines) < 2:
        return None
    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    content_lines = norm_content.splitlines()
    for i, line in enumerate(content_lines):
        if line.strip() == first:
            for j in range(i + 1, min(i + len(old_lines) + 5, len(content_lines))):
                if content_lines[j].strip() == last:
                    original_lines = content.splitlines()
                    matched = "\n".join(original_lines[i:j + 1])
                    result = content.replace(matched, new, 1)
                    return result, 1
    return None


def _strategy_block_anchor(content, norm_content, old, norm_old, new, replace_all):
    old_lines = norm_old.splitlines()
    if len(old_lines) < 3:
        return None
    first = old_lines[0].strip()
    last = old_lines[-1].strip()
    content_lines = content.splitlines()
    for i, line in enumerate(content_lines):
        if line.strip() == first:
            expected_end = i + len(old_lines) - 1
            if expected_end < len(content_lines) and content_lines[expected_end].strip() == last:
                block = content_lines[i:expected_end + 1]
                sim = _block_similarity(old_lines, block)
                if sim > 0.6:
                    matched = "\n".join(block)
                    return content.replace(matched, new, 1), 1
    return None


def _strategy_context_aware(content, norm_content, old, norm_old, new, replace_all):
    old_lines = norm_old.splitlines()
    content_lines = content.splitlines()
    if len(old_lines) < 2 or len(content_lines) < len(old_lines):
        return None

    best_score = 0.0
    best_start = -1
    window = len(old_lines)

    for i in range(len(content_lines) - window + 1):
        block = content_lines[i:i + window]
        score = _block_similarity(old_lines, block)
        if score > best_score:
            best_score = score
            best_start = i

    if best_score >= 0.5 and best_start >= 0:
        matched = "\n".join(content_lines[best_start:best_start + window])
        return content.replace(matched, new, 1), 1

    return None


def _block_similarity(a_lines: list[str], b_lines: list[str]) -> float:
    if not a_lines or not b_lines:
        return 0.0
    matches = sum(
        1 for al, bl in zip(a_lines, b_lines)
        if SequenceMatcher(None, al.strip(), bl.strip()).ratio() > 0.8
    )
    return matches / max(len(a_lines), len(b_lines))


def _replace_by_trimmed_lines(content: str, old: str, new: str, replace_all: bool):
    """Find the actual content block matching old (by trimmed lines) and replace."""
    old_lines = [line.strip() for line in old.splitlines()]
    content_lines = content.splitlines()

    for i in range(len(content_lines) - len(old_lines) + 1):
        block = content_lines[i:i + len(old_lines)]
        if all(cl.strip() == ol for cl, ol in zip(block, old_lines)):
            matched = "\n".join(content_lines[i:i + len(old_lines)])
            result = content.replace(matched, new, 1)
            return result, 1

    return None
