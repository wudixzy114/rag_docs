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
    obj = _parse_container(text, "[", "]")
    return obj if isinstance(obj, list) else None


def parse_json_object(text: str) -> dict | None:
    """Extract the outermost JSON object. None if unparseable."""
    obj = _parse_container(text, "{", "}")
    return obj if isinstance(obj, dict) else None


def _parse_container(text: str, opener: str, closer: str):
    """Parse common model JSON defects without guessing document content.

    ``strict=False`` accepts literal newlines/tabs inside Markdown strings, the
    dominant production failure. A second pass removes trailing commas outside
    strings. Balanced scanning avoids the old first/last bracket behavior when a
    model surrounds JSON with prose containing brackets.
    """
    source = _strip_fences(text)
    for candidate in _balanced_candidates(source, opener, closer):
        for value in (candidate, _remove_trailing_commas(candidate)):
            try:
                return json.loads(value, strict=False)
            except (json.JSONDecodeError, ValueError):
                continue
    return None


def _balanced_candidates(text: str, opener: str, closer: str):
    start = None
    depth = 0
    in_string = False
    escaped = False
    for index, char in enumerate(text):
        if in_string:
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            continue
        if char == '"':
            in_string = True
        elif char == opener:
            if depth == 0:
                start = index
            depth += 1
        elif char == closer and depth:
            depth -= 1
            if depth == 0 and start is not None:
                yield text[start:index + 1]
                start = None


def _remove_trailing_commas(text: str) -> str:
    out: list[str] = []
    in_string = False
    escaped = False
    index = 0
    while index < len(text):
        char = text[index]
        if in_string:
            out.append(char)
            if escaped:
                escaped = False
            elif char == "\\":
                escaped = True
            elif char == '"':
                in_string = False
            index += 1
            continue
        if char == '"':
            in_string = True
        if char == ",":
            lookahead = index + 1
            while lookahead < len(text) and text[lookahead].isspace():
                lookahead += 1
            if lookahead < len(text) and text[lookahead] in "}]":
                index += 1
                continue
        out.append(char)
        index += 1
    return "".join(out)


def _strip_fences(text: str) -> str:
    s = (text or "").strip()
    s = re.sub(r"^```(?:json)?|```$", "", s, flags=re.MULTILINE).strip()
    return s
