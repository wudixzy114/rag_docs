"""Reversible sensitive-content masking.

The gateway's content-safety filter hard-rejects (400 FAILED_PRECONDITION) any
request containing MAC addresses / internal IPs / phone numbers / credential
tokens. So we MUST mask before sending. But the knowledge base is used INTERNALLY
(company chat groups) where these values should be visible — so masking must be
REVERSIBLE: mask for the gateway, restore on export.

Mechanism:
- Each sensitive value → a deterministic placeholder `〔KIND:hash〕` where hash is
  derived from the value. Same value always maps to the same token, so caching,
  dedup and cross-file aggregation are unaffected (the masked text is stable).
- The Redactor records value↔token both ways. `restore()` swaps tokens back to
  the real values on export. The map is also persisted (redaction_map.json) as an
  audit trail.
- Uses full-width brackets 〔〕 and a hex suffix so a placeholder never collides
  with real content and survives round-tripping through the model verbatim.

User decision (2026-07-14): restore EVERYTHING internally, incl. credentials — no
denylist. If that ever changes, `restore()` is the single choke point to gate.
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

# Detection patterns, each tagged with a KIND label used in the placeholder.
_PATTERNS = [
    ("MAC", re.compile(r"\b(?:[0-9A-Fa-f]{2}[:-]){5}[0-9A-Fa-f]{2}\b")),
    ("IPV6", re.compile(r"\bfe80::[0-9A-Fa-f:]{2,}\b")),
    ("IP", re.compile(
        r"\b(?:10\.\d{1,3}\.\d{1,3}\.\d{1,3}"
        r"|172\.(?:1[6-9]|2\d|3[01])\.\d{1,3}\.\d{1,3}"
        r"|192\.168\.\d{1,3}\.\d{1,3})\b")),
    ("PHONE", re.compile(r"\b1[3-9]\d{9}\b")),          # CN mobile numbers
    ("EMAIL", re.compile(r"(?i)\b[A-Z0-9._%+\-]+@[A-Z0-9.\-]+\.[A-Z]{2,}\b")),
    ("BEARER", re.compile(r"(?i)(?<=bearer )[A-Za-z0-9._\-]{8,}")),
    ("KEY", re.compile(r"(?i)(?<=api[_-]key[=: ])[A-Za-z0-9._\-]{12,}")),
    ("PASSWORD", re.compile(
        r"(?i)(?:\b[A-Za-z0-9_-]*(?:password|passwd|pwd)|密码)\s*[:=：]\s*"
        r"[^\s,，;；'\"`]{4,}")),
    ("SECRET", re.compile(
        r"(?i)\b(?:client[_-]?secret|access[_-]?token|secret[_-]?key)\s*[:=]\s*"
        r"[^\s,，;；'\"`]{8,}")),
    ("HEX32", re.compile(r"\b[0-9a-f]{32}\b")),          # gateway-key-shaped tokens
]


class Redactor:
    """Stateful, reversible masker. One instance per run; thread-safe so parallel
    doc workers share a single value↔token map (a value seen in two docs gets the
    same token, keeping cross-file dedup stable)."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._to_token: dict[str, str] = {}
        self._to_value: dict[str, str] = {}

    def _token(self, kind: str, value: str) -> str:
        h = hashlib.sha256(value.encode("utf-8")).hexdigest()[:6]
        return f"〔{kind}:{h}〕"

    def mask(self, text: str) -> str:
        """Replace every sensitive value with its deterministic placeholder,
        recording the mapping. Idempotent: masking already-masked text is a no-op
        (placeholders don't match the patterns)."""
        if not text:
            return text
        s = text
        for kind, pat in _PATTERNS:
            def _sub(m: re.Match) -> str:
                val = m.group(0)
                tok = self._token(kind, val)
                with self._lock:
                    self._to_token[val] = tok
                    self._to_value[tok] = val
                return tok
            s = pat.sub(_sub, s)
        return s

    def restore(self, text: str) -> str:
        """Swap every known placeholder back to its real value (export path)."""
        if not text:
            return text
        s = text
        with self._lock:
            items = list(self._to_value.items())
        for tok, val in items:
            if tok in s:
                s = s.replace(tok, val)
        return s

    def mapping(self) -> dict[str, str]:
        """token → real value, for the audit sidecar (redaction_map.json)."""
        with self._lock:
            return dict(self._to_value)

    def load(self, path: Path) -> None:
        path = Path(path)
        if not path.is_file():
            return
        try:
            values = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return
        if not isinstance(values, dict):
            return
        with self._lock:
            for token, value in values.items():
                self._to_value[str(token)] = str(value)
                self._to_token[str(value)] = str(token)

    def save(self, path: Path) -> None:
        path = Path(path)
        with self._lock:
            values = dict(self._to_value)
        if not values:
            return
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(values, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(path)
        try:
            path.chmod(0o600)
        except OSError:
            pass


# A process-wide default redactor so the mask (extract/vision) and restore
# (export) paths share one map without threading it through every call.
_DEFAULT = Redactor()


def mask(text: str) -> str:
    return _DEFAULT.mask(text)


def restore(text: str) -> str:
    return _DEFAULT.restore(text)


def mapping() -> dict[str, str]:
    return _DEFAULT.mapping()


def load_mapping(path: Path) -> None:
    _DEFAULT.load(path)


def save_mapping(path: Path) -> None:
    _DEFAULT.save(path)


_TOKEN_RE = re.compile(r"〔[A-Z0-9_]+:[0-9a-f]{6}〕")


def unresolved_tokens(text: str) -> list[str]:
    """Placeholders that cannot be restored by the currently loaded mapping."""
    tokens = set(_TOKEN_RE.findall(text or ""))
    known = set(mapping())
    return sorted(tokens - known)
