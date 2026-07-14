"""Manifest — per-document processing state + pin registry.

One JSON file (`output/manifest.json`) tracks, per source document (keyed by
topic folder):
- source_sha: sha256 of the source file bytes → drives incremental re-run
  (unchanged sha + unchanged pipeline version = skip).
- status: pending | running | done | failed
- counts: extracted / passed / rejected units, for the progress view
- pinned: once the user approves a doc's output, it's pinned; a re-run SKIPS
  pinned docs so approved work is never overwritten.
- error: last failure message, for the retry UI.

The manifest is the single source of truth the dashboard reads and the
orchestrator writes. Writes are atomic (tmp+replace) so a crash never corrupts it.
"""
from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field, asdict
from pathlib import Path

# Bump when pipeline logic changes in a way that should force re-processing even
# on unchanged sources (new prompt version, new stage). Combined with source_sha
# into the skip decision.
PIPELINE_VERSION = "v1"


@dataclass
class DocState:
    topic: str
    source_sha: str = ""
    status: str = "pending"          # pending|running|done|failed
    pinned: bool = False
    extracted: int = 0
    passed: int = 0
    rejected: int = 0
    needs_review: int = 0
    sop_count: int = 0
    error: str = ""
    pipeline_version: str = PIPELINE_VERSION

    def to_dict(self) -> dict:
        return asdict(self)


class Manifest:
    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.Lock()
        self._docs: dict[str, DocState] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.is_file():
            return
        try:
            data = json.loads(self.path.read_text("utf-8"))
        except (json.JSONDecodeError, ValueError, OSError):
            return
        for topic, d in (data.get("docs") or {}).items():
            self._docs[topic] = DocState(**{**{"topic": topic}, **d})

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": {t: s.to_dict() for t, s in self._docs.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.path)

    def get(self, topic: str) -> DocState | None:
        with self._lock:
            return self._docs.get(topic)

    def all(self) -> list[DocState]:
        with self._lock:
            return list(self._docs.values())

    def upsert(self, state: DocState) -> None:
        with self._lock:
            self._docs[state.topic] = state
            self._save()

    def update(self, topic: str, **fields) -> DocState:
        with self._lock:
            st = self._docs.get(topic) or DocState(topic=topic)
            for k, v in fields.items():
                setattr(st, k, v)
            self._docs[topic] = st
            self._save()
            return st

    def set_pinned(self, topic: str, pinned: bool) -> None:
        self.update(topic, pinned=pinned)

    def should_skip(self, topic: str, source_sha: str) -> bool:
        """Skip if pinned, OR already done with the same source + pipeline version."""
        st = self.get(topic)
        if not st:
            return False
        if st.pinned:
            return True
        return (st.status == "done" and st.source_sha == source_sha
                and st.pipeline_version == PIPELINE_VERSION)
