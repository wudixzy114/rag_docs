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
PIPELINE_VERSION = "v4"


@dataclass
class DocState:
    topic: str
    source_sha: str = ""
    status: str = "pending"          # pending|running|done|partial|failed|interrupted|awaiting_review|excluded
    pinned: bool = False
    extracted: int = 0
    passed: int = 0
    rejected: int = 0
    needs_review: int = 0
    sop_count: int = 0
    error: str = ""
    error_code: str = ""
    retryable: bool = False
    current_stage: str = ""
    last_completed_stage: str = ""
    preflight_status: str = "pending"  # pending|included|awaiting_review|excluded
    preflight_flags: list[str] = field(default_factory=list)
    decision_id: str = ""
    attempts: int = 0
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

    def reload(self) -> None:
        """Re-read the on-disk manifest into memory.

        The dashboard server holds one long-lived Manifest, but a CLI run in a
        SEPARATE process appends new docs to manifest.json. Without this the
        server keeps serving its startup snapshot, so an appended batch never
        enlarges the total and the progress bar sticks at 100%. Callers on the
        read path (api_state) invoke this so cross-process writes are visible.
        """
        with self._lock:
            self._docs = {}
            self._load()

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"docs": {t: s.to_dict() for t, s in self._docs.items()}}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

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

    def bulk_update(self, changes: dict[str, dict]) -> None:
        """Apply a migration atomically with one manifest rewrite."""
        with self._lock:
            for topic, fields in changes.items():
                state = self._docs.get(topic) or DocState(topic=topic)
                for key, value in fields.items():
                    setattr(state, key, value)
                self._docs[topic] = state
            if changes:
                self._save()

    def set_pinned(self, topic: str, pinned: bool) -> None:
        self.update(topic, pinned=pinned)

    def recover_interrupted(self) -> list[str]:
        """Turn stale running states into explicit, retryable interruptions."""
        with self._lock:
            topics = []
            for topic, state in self._docs.items():
                if state.status != "running":
                    continue
                state.status = "interrupted"
                state.error_code = "interrupted"
                state.error = "上次进程在阶段完成前中断，可从缓存安全重试"
                state.retryable = True
                topics.append(topic)
            if topics:
                self._save()
            return topics

    def should_skip(self, topic: str, source_sha: str) -> bool:
        """Skip if pinned, OR already done with the same source + pipeline version."""
        st = self.get(topic)
        if not st:
            return False
        if st.pinned:
            return True
        return (st.status == "done" and st.source_sha == source_sha
                and st.pipeline_version == PIPELINE_VERSION)
