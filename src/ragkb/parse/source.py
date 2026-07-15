"""Load common document formats into a Markdown-like intermediate text."""
from __future__ import annotations

import html
import hashlib
import re
import zipfile
from html.parser import HTMLParser
from pathlib import Path
from xml.etree import ElementTree as ET


SUPPORTED_EXTENSIONS = {".md", ".markdown", ".txt", ".html", ".htm", ".docx", ".pdf"}
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".gif", ".webp"}


def source_bundle_sha(path: Path) -> str:
    """Hash a source plus sibling image assets used by document references."""
    path = Path(path)
    digest = hashlib.sha256()
    digest.update(path.name.encode("utf-8"))
    digest.update(b"\0")
    digest.update(path.read_bytes())
    for asset in sorted(p for p in path.parent.rglob("*")
                        if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS):
        digest.update(b"\0")
        digest.update(asset.relative_to(path.parent).as_posix().encode("utf-8"))
        digest.update(b"\0")
        digest.update(asset.read_bytes())
    return digest.hexdigest()


def load_source(path: Path) -> str:
    """Return source content as Markdown-like text.

    The parser downstream only needs headings, paragraphs and code/table text;
    preserving those semantics is more robust than branching the whole pipeline
    by file type. PDF support uses pypdf when installed and fails explicitly when
    a scanned/image-only PDF has no extractable text.
    """
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"unsupported source format: {suffix or '<none>'}")
    if suffix in {".md", ".markdown", ".txt"}:
        return _decode_text(path.read_bytes())
    if suffix in {".html", ".htm"}:
        parser = _HTMLToMarkdown()
        parser.feed(_decode_text(path.read_bytes()))
        return parser.text()
    if suffix == ".docx":
        return _docx_to_markdown(path)
    return _pdf_to_text(path)


def _decode_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "gb18030"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", "replace")


class _HTMLToMarkdown(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self._lines: list[str] = []
        self._buf: list[str] = []
        self._heading = 0
        self._pre = False
        self._list_depth = 0

    def _flush(self, prefix: str = "") -> None:
        value = html.unescape("".join(self._buf)).strip()
        self._buf.clear()
        if value:
            self._lines.append(prefix + re.sub(r"[ \t]+", " ", value))

    def handle_starttag(self, tag: str, attrs) -> None:
        tag = tag.lower()
        if re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self._heading = int(tag[1])
        elif tag == "pre":
            self._flush()
            self._pre = True
            self._lines.append("```")
        elif tag in {"ul", "ol"}:
            self._list_depth += 1
        elif tag == "li":
            self._flush()
        elif tag == "br":
            self._buf.append("\n")

    def handle_endtag(self, tag: str) -> None:
        tag = tag.lower()
        if re.fullmatch(r"h[1-6]", tag):
            self._flush("#" * self._heading + " ")
            self._heading = 0
        elif tag == "pre":
            self._flush()
            self._lines.append("```")
            self._pre = False
        elif tag in {"p", "div", "section", "article", "tr"}:
            self._flush()
        elif tag == "li":
            self._flush("  " * max(0, self._list_depth - 1) + "- ")
        elif tag in {"ul", "ol"}:
            self._list_depth = max(0, self._list_depth - 1)
        elif tag in {"td", "th"}:
            self._buf.append(" | ")

    def handle_data(self, data: str) -> None:
        self._buf.append(data if self._pre else re.sub(r"\s+", " ", data))

    def text(self) -> str:
        self._flush()
        return "\n\n".join(line for line in self._lines if line.strip()).strip()


def _docx_to_markdown(path: Path) -> str:
    ns = {"w": "http://schemas.openxmlformats.org/wordprocessingml/2006/main"}
    try:
        with zipfile.ZipFile(path) as archive:
            root = ET.fromstring(archive.read("word/document.xml"))
    except (zipfile.BadZipFile, KeyError, ET.ParseError) as exc:
        raise ValueError(f"invalid DOCX: {path}") from exc

    lines: list[str] = []
    for node in root.findall(".//w:body/*", ns):
        kind = node.tag.rsplit("}", 1)[-1]
        if kind == "p":
            text = "".join(t.text or "" for t in node.findall(".//w:t", ns)).strip()
            if not text:
                continue
            style = node.find("./w:pPr/w:pStyle", ns)
            style_name = style.get(f"{{{ns['w']}}}val", "") if style is not None else ""
            match = re.search(r"(?:Heading|标题)\s*([1-6])", style_name, re.I)
            lines.append(("#" * int(match.group(1)) + " " if match else "") + text)
        elif kind == "tbl":
            for row in node.findall("./w:tr", ns):
                cells = ["".join(t.text or "" for t in cell.findall(".//w:t", ns)).strip()
                         for cell in row.findall("./w:tc", ns)]
                if any(cells):
                    lines.append("| " + " | ".join(cells) + " |")
    text = "\n\n".join(lines).strip()
    if not text:
        raise ValueError(f"DOCX contains no readable text: {path}")
    return text


def _pdf_to_text(path: Path) -> str:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - dependency installation issue
        raise RuntimeError("PDF input requires the 'pypdf' dependency") from exc
    try:
        pages = [page.extract_text() or "" for page in PdfReader(str(path)).pages]
    except Exception as exc:  # pypdf exposes several backend-specific errors
        raise ValueError(f"invalid PDF: {path}") from exc
    text = "\n\n".join(p.strip() for p in pages if p.strip())
    if not text:
        raise ValueError(f"PDF has no extractable text (OCR required): {path}")
    return text
