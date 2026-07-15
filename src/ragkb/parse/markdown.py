"""Fence-aware Markdown → Document parser.

Turns one `原始文档.md` into a `Document` (section tree + image nodes). The two
correctness-critical behaviors, both grounded in the real material:

1. **Fence awareness.** A line matching `^#{1,6} ` inside a ``` code fence is a
   code comment, not a heading (`# 安装依赖`, `# count`, `# your train code`).
   Splitting on it would scramble sections — exactly the "错乱" the user forbids.
   The parser tracks fence open/close and only treats headings OUTSIDE fences.

2. **OCR-block lifting.** An image `![](images/img-NN.png)` is followed by
   `<!-- ocr-source: PATH -->` + a <details><summary>…</summary><pre>OCR</pre>
   </details> block. We parse the image into an `Image` node (recording the weak
   inline OCR as a hint only) and STRIP the raw OCR scaffolding out of the body
   text, so the body stays clean prose. The authoritative text comes later from
   the vision layer reading the real image file.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

from ragkb.parse.model import Document, Image, Section

_HEADING_RE = re.compile(r"^(#{1,6})\s+(.*\S)\s*$")
_FENCE_RE = re.compile(r"^\s*(`{3,}|~{3,})")
_IMG_RE = re.compile(r"!\[[^\]]*\]\(([^)]+)\)")
_OCR_SOURCE_RE = re.compile(r"<!--\s*ocr-source:\s*(.+?)\s*-->")


def parse_document(md_path: Path, topic: str | None = None) -> Document:
    """Parse a markdown file into a Document. `topic` defaults to the parent
    folder name (the aggregation boundary)."""
    md_path = Path(md_path)
    raw = md_path.read_bytes()
    text = raw.decode("utf-8")
    source_sha = hashlib.sha256(raw).hexdigest()
    topic = topic or md_path.parent.name
    base_dir = md_path.parent

    lines = text.splitlines()
    root = Section(level=0, title="__root__")
    stack: list[Section] = [root]
    # Per-heading child counters, for dotted section ids (3, 3.1, 3.2...).
    counter: dict[int, int] = {}

    in_fence = False
    fence_marker = ""
    i = 0
    n = len(lines)
    # Body accumulator for the current section.
    buf: list[str] = []

    def flush_body(sec: Section) -> None:
        # Trim leading/trailing blank lines; keep internal structure.
        txt = "\n".join(buf).strip("\n")
        sec.body = (sec.body + "\n" + txt).strip("\n") if sec.body else txt
        buf.clear()

    while i < n:
        line = lines[i]

        # Fence tracking: a ``` / ~~~ toggles fence state. Inside a fence, nothing
        # is a heading or an image directive — it's literal code.
        fm = _FENCE_RE.match(line)
        if fm:
            marker = fm.group(1)[0] * 3
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif line.strip().startswith(fence_marker):
                in_fence = False
                fence_marker = ""
            buf.append(line)
            i += 1
            continue

        if in_fence:
            buf.append(line)
            i += 1
            continue

        # OCR block: `<!-- ocr-source: PATH -->` then <details>…<pre>OCR</pre></details>.
        # Attach the OCR text to the most recent image on the current section, and
        # consume the whole block so it never lands in body prose.
        mo = _OCR_SOURCE_RE.search(line)
        if mo:
            ocr_path = mo.group(1).strip()
            block_end, ocr_text = _consume_ocr_block(lines, i)
            _attach_ocr(stack[-1], ocr_path, ocr_text)
            i = block_end
            continue

        # Heading (outside fences): close deeper sections, open a new one.
        mh = _HEADING_RE.match(line)
        if mh:
            flush_body(stack[-1])
            level = len(mh.group(1))
            title = mh.group(2).strip()
            # Pop to parent: keep sections with level < this one.
            while stack and stack[-1].level >= level:
                stack.pop()
            parent = stack[-1] if stack else root
            counter_key = id(parent)
            # Dotted sid based on position under parent.
            idx = _bump(counter, level, parent)
            sid = f"{parent.sid}.{idx}" if parent.sid else str(idx)
            heading_path = [*parent.heading_path, parent.title] if parent.level > 0 else []
            sec = Section(level=level, title=title, heading_path=heading_path, sid=sid)
            parent.children.append(sec)
            stack.append(sec)
            i += 1
            continue

        # Image directive (outside fences): record an Image node on current section.
        for m in _IMG_RE.finditer(line):
            rel = m.group(1).strip()
            _add_image(stack[-1], rel, base_dir)
        buf.append(line)
        i += 1

    flush_body(stack[-1])

    title = root.children[0].title if root.children and root.children[0].level == 1 else topic
    return Document(path=md_path, topic=topic, title=title, root=root, source_sha=source_sha)


def _bump(counter: dict, level: int, parent: Section) -> int:
    key = (id(parent),)
    counter[key] = counter.get(key, 0) + 1
    return counter[key]


def _consume_ocr_block(lines: list[str], start: int) -> tuple[int, str]:
    """From the `<!-- ocr-source -->` line, consume through </details>, returning
    (index_after_block, ocr_text_inside_<pre>). Tolerant if </details> is absent
    (stops at a blank line run)."""
    i = start + 1
    n = len(lines)
    ocr_lines: list[str] = []
    in_pre = False
    while i < n:
        ln = lines[i]
        low = ln.strip().lower()
        # A line may pack <pre>…</pre></details><tail> together. Detect an inline
        # </details> on THIS line (after any </pre>) so a glued trailing image
        # directive is handed back to the main loop rather than skipped.
        if not in_pre and "<pre" in low and "</details>" in low:
            after = ln.split(">", 1)[1] if ">" in ln else ""
            if "</pre>" in after:
                ocr_lines.append(after.split("</pre>")[0])
            _det = re.split(r"</details>", ln, maxsplit=1, flags=re.I)
            tail = _det[1] if len(_det) > 1 else ""
            ocr_text = _unescape("\n".join(ocr_lines)).strip()
            if tail.strip():
                lines[i] = tail
                return i, ocr_text
            return i + 1, ocr_text
        if "<pre" in low:
            in_pre = True
            after = ln.split(">", 1)[1] if ">" in ln else ""
            if "</pre>" in after:
                ocr_lines.append(after.split("</pre>")[0])
                in_pre = False
            elif after:
                ocr_lines.append(after)
            i += 1
            continue
        if in_pre:
            if "</pre>" in low:
                ocr_lines.append(ln.split("</pre>")[0])
                in_pre = False
                i += 1
                continue
            ocr_lines.append(ln)
            i += 1
            continue
        if "</details>" in low:
            # Preserve any content glued AFTER </details> on the same line
            # (e.g. `</details>![](images/img-10.png)` — the next image directive).
            # Returning i+1 would skip that whole line and silently drop the image.
            # Rewrite the line to just its trailing part and hand the SAME index
            # back so the main loop reprocesses it (image scan, heading, etc.).
            parts = re.split(r"</details>", ln, maxsplit=1, flags=re.I)
            tail = parts[1] if len(parts) > 1 else ""
            ocr_text = _unescape("\n".join(ocr_lines)).strip()
            if tail.strip():
                lines[i] = tail
                return i, ocr_text
            return i + 1, ocr_text
        i += 1
    return i, _unescape("\n".join(ocr_lines)).strip()


def _unescape(s: str) -> str:
    return (s.replace("&lt;", "<").replace("&gt;", ">")
             .replace("&quot;", '"').replace("&#39;", "'").replace("&amp;", "&"))


def _add_image(sec: Section, rel: str, base_dir: Path) -> Image:
    abs_path = (base_dir / rel).resolve()
    img = Image(rel_path=rel, abs_path=abs_path, exists=abs_path.is_file())
    sec.images.append(img)
    return img


def _attach_ocr(sec: Section, ocr_path: str, ocr_text: str) -> None:
    """Attach OCR text to the matching image (by path) on this section, else the
    most recent image; if none exists yet, create a node for the referenced path."""
    for img in reversed(sec.images):
        if img.rel_path == ocr_path or img.rel_path.endswith(ocr_path) or ocr_path.endswith(img.rel_path):
            img.inline_ocr = ocr_text
            return
    if sec.images:
        sec.images[-1].inline_ocr = ocr_text
