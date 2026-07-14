"""Parsed-document data model.

A source `原始文档.md` becomes a `Document`: an ordered tree of `Section`s (by
heading level) each carrying its own body text and the `Image`s that appear
within it. This is the substrate every later stage consumes — classification
labels sections, QA extraction reads section body + images, SOP cleaning
serializes a section subtree back to Markdown.

Design notes grounded in the real material:
- Headings are `#`..`####`. Lines that LOOK like headings but sit inside a ```
  fence are code comments (`# 安装依赖`, `# count`), NOT headings — the parser is
  fence-aware, so they stay in body text and never split a section.
- Images arrive as `![](images/img-NN.png)` followed by an inline OCR block
  (`<!-- ocr-source: PATH -->` + <details><pre>…</pre></details>). The pre-baked
  OCR is LOW QUALITY, so we keep it only as a weak hint; the authoritative text
  is produced later by the vision layer reading `abs_path`.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Image:
    """One image reference inside a section."""
    rel_path: str                 # as written in the markdown, e.g. images/img-01.png
    abs_path: Path                # resolved absolute path to the real image file
    inline_ocr: str = ""          # pre-baked OCR — LOW confidence, hint only
    vision_text: str = ""         # authoritative transcription (filled by vision layer)
    exists: bool = True           # abs_path resolved to a real file
    vision_failed: bool = False   # vision read failed (quota/error) — image left unread

    @property
    def best_text(self) -> str:
        """Authoritative text if vision has run, else the weak OCR hint."""
        return self.vision_text.strip() or self.inline_ocr.strip()

    _SUFFIX_MEDIA = {
        ".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg",
        ".gif": "image/gif", ".webp": "image/webp",
    }

    def data_bytes(self) -> bytes:
        return self.abs_path.read_bytes()

    def media_type(self) -> str:
        return self._SUFFIX_MEDIA.get(self.abs_path.suffix.lower(), "image/png")


@dataclass
class Section:
    """A heading and everything under it up to the next same-or-higher heading.

    `body` is the section's own prose (excluding descendant sections and the raw
    OCR blocks, which are lifted into `images`). `heading_path` is the chain of
    ancestor titles, used both as breadcrumbs for the LLM and as provenance.
    """
    level: int                    # 1..6 (0 = synthetic document root)
    title: str
    heading_path: list[str] = field(default_factory=list)
    body: str = ""
    images: list[Image] = field(default_factory=list)
    children: list["Section"] = field(default_factory=list)
    # Stable id within a document: dotted section numbers by position, e.g. "3.1".
    sid: str = ""

    def iter_sections(self):
        """Depth-first over self + descendants (document order)."""
        yield self
        for c in self.children:
            yield from c.iter_sections()

    @property
    def full_title(self) -> str:
        return " / ".join([*self.heading_path, self.title]) if self.heading_path else self.title

    def all_images(self) -> list[Image]:
        out = list(self.images)
        for c in self.children:
            out.extend(c.all_images())
        return out


@dataclass
class Document:
    """One source FAQ file. `topic` is the containing folder name (01_… 10_…),
    which is ALSO the cross-file aggregation boundary."""
    path: Path                    # absolute path to 原始文档.md
    topic: str                    # containing folder name, the topic/职责 boundary
    title: str                    # document H1 (or folder name if none)
    root: Section                 # synthetic level-0 root; real sections are children
    source_sha: str = ""          # sha256 of file bytes, for idempotency

    def iter_sections(self):
        """All real sections (skips the synthetic root)."""
        for c in self.root.children:
            yield from c.iter_sections()

    def all_images(self) -> list[Image]:
        return self.root.all_images()
