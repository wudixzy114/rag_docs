"""Normalize highly variable source structure before LLM classification."""
from __future__ import annotations

from dataclasses import replace

from ragkb.parse.model import Section


def split_oversize_sections(sections: list[Section], max_chars: int = 14000) -> list[Section]:
    """Split long prose on paragraph boundaries while preserving code fences."""
    output: list[Section] = []
    for section in sections:
        if len(section.body) <= max_chars:
            output.append(section)
            continue
        chunks = _chunks(section.body, max_chars)
        for index, body in enumerate(chunks, 1):
            output.append(replace(
                section,
                title=f"{section.title}（{index}/{len(chunks)}）",
                body=body,
                images=section.images if index == 1 else [],
                children=[],
                sid=f"{section.sid}.part{index}",
            ))
    return output


def _chunks(text: str, max_chars: int) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    in_fence = False
    for line in text.splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            in_fence = not in_fence
        if not line.strip() and not in_fence and current:
            blocks.append("\n".join(current).strip())
            current = []
        else:
            current.append(line)
    if current:
        blocks.append("\n".join(current).strip())

    chunks: list[str] = []
    buf: list[str] = []
    size = 0
    for block in blocks:
        extra = len(block) + (2 if buf else 0)
        if buf and size + extra > max_chars:
            chunks.append("\n\n".join(buf))
            buf, size = [], 0
        # An indivisible code block may exceed the budget; keeping its fences
        # intact is safer than sending structurally corrupted fragments.
        buf.append(block)
        size += len(block) + (2 if len(buf) > 1 else 0)
    if buf:
        chunks.append("\n\n".join(buf))
    return chunks or [text]
