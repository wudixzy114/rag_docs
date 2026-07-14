"""Tolerant JSON extraction from LLM output.

Mirrors magnus-lens verify.py `_parse_batch_verdicts`: strip ``` fences, find the
outermost [...] or {...}, parse. Returns a default on any failure so callers
degrade gracefully (fail-closed downstream) rather than crashing a whole run on
one malformed response.
"""
from __future__ import annotations

import json
import re


def parse_json_array(text: str) -> list | None:
    """Extract the outermost JSON array. None if unparseable."""
    s = _strip_fences(text)
    lo, hi = s.find("["), s.rfind("]")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(s[lo:hi + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, list) else None


def parse_json_object(text: str) -> dict | None:
    """Extract the outermost JSON object. None if unparseable."""
    s = _strip_fences(text)
    lo, hi = s.find("{"), s.rfind("}")
    if lo == -1 or hi == -1 or hi <= lo:
        return None
    try:
        obj = json.loads(s[lo:hi + 1])
    except (json.JSONDecodeError, ValueError):
        return None
    return obj if isinstance(obj, dict) else None


def _strip_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    return s
