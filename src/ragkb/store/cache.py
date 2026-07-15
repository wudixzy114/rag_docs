"""Content-addressed cache for expensive, deterministic work (vision reads, LLM
extractions). Keyed by a caller-chosen string (usually a hash of the exact
inputs: file bytes + model + prompt version), so a re-run with unchanged inputs
is an instant disk hit and only changed items re-run.

Deliberately simple: one JSON file per key under a namespace subdir. No TTL —
invalidation is by key change (bump a prompt/parser version → new key). Never
keyed on time/random, so runs are reproducible and resumable.
"""
from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any


class Cache:
    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._write_lock = threading.Lock()

    def _path(self, namespace: str, key: str) -> Path:
        safe = hashlib.sha256(key.encode("utf-8")).hexdigest()[:32]
        return self.root / namespace / f"{safe}.json"

    def get(self, namespace: str, key: str) -> Any | None:
        p = self._path(namespace, key)
        if not p.is_file():
            return None
        try:
            return json.loads(p.read_text("utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return None

    def put(self, namespace: str, key: str, value: Any) -> None:
        p = self._path(namespace, key)
        with self._write_lock:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = p.with_suffix(".tmp")
            tmp.write_text(json.dumps(value, ensure_ascii=False, indent=0), "utf-8")
            tmp.replace(p)      # atomic: a crash mid-write never leaves a half file

    def has(self, namespace: str, key: str) -> bool:
        return self._path(namespace, key).is_file()


def key_for(*parts: str) -> str:
    """Stable composite key from ordered parts (e.g. sha + model + prompt_ver)."""
    return "\x1f".join(parts)
