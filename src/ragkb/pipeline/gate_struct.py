"""Layer 1 — deterministic structural gate (fail-closed).

This gate NEVER judges content quality — that's Layer 2's job (a strong model).
It only catches STRUCTURAL corruption, the failure modes the user explicitly
forbids (截断/错乱) and which an LLM reviewer is unreliable at spotting (it tends
to "read past" a truncation). Everything here is free and 100% deterministic, so
it loses no information: a short-but-correct answer passes; only a broken one fails.

Checks:
- truncated: finish_reason==length upstream → hard fail (must re-run bigger).
- empty: blank query or answer.
- unbalanced code fences (odd number of ``` → a fence was cut mid-block).
- unterminated inline markup that signals a cut (dangling ``` at end).
- query == answer (degenerate).
- CSV-safety is guaranteed at export by QUOTE_ALL, so we don't reject on commas.
"""
from __future__ import annotations

import re

from ragkb.pipeline.units import QAUnit, SOPUnit

_MIN_ANSWER = 1        # non-empty; we do NOT impose a length floor (would lose info)


def _fence_balanced(text: str) -> bool:
    opened: tuple[str, int] | None = None
    for line in (text or "").splitlines():
        match = re.match(r"^\s*(`{3,}|~{3,})", line)
        if not match:
            continue
        marker = match.group(1)
        current = (marker[0], len(marker))
        if opened is None:
            opened = current
        elif current[0] == opened[0] and current[1] >= opened[1]:
            opened = None
    return opened is None


def gate_qa(unit: QAUnit) -> QAUnit:
    reasons = []
    if unit.truncated:
        reasons.append("truncated(finish_reason=length)")
    if not unit.query.strip():
        reasons.append("empty_query")
    if not unit.answer.strip():
        reasons.append("empty_answer")
    if unit.query.strip() and unit.query.strip() == unit.answer.strip():
        reasons.append("query_equals_answer")
    if not _fence_balanced(unit.answer):
        reasons.append("unbalanced_code_fence")
    unit.struct_ok = not reasons
    unit.struct_reason = ";".join(reasons)
    return unit


def gate_sop(unit: SOPUnit) -> SOPUnit:
    reasons = []
    if unit.truncated:
        reasons.append("truncated(finish_reason=length)")
    if not unit.markdown.strip():
        reasons.append("empty_markdown")
    if not _fence_balanced(unit.markdown):
        reasons.append("unbalanced_code_fence")
    if not unit.entry_questions:
        reasons.append("no_entry_questions")
    if unit.markdown.strip() and not re.search(r"^#{1,6}\s+\S", unit.markdown, re.MULTILINE):
        reasons.append("no_markdown_heading")
    unit.struct_ok = not reasons
    unit.struct_reason = ";".join(reasons)
    return unit
