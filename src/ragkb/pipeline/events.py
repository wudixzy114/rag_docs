"""In-process event bus + cross-process journal for live observability.

The orchestrator publishes progress events (doc started, section extracted, unit
gated, doc done, error); the dashboard's SSE endpoint subscribes and streams them
to the browser. Also retains a bounded history so a late-connecting browser can
replay recent state, and the CLI can print a running log.

Cross-process: a run started from the CLI (`ragkb run`) lives in a DIFFERENT
process than the dashboard server (`ragkb serve`), so an in-memory bus alone can
never feed that browser. When constructed with a `journal_path`, every published
event is also appended as one JSON line to that file; the dashboard tails the
file by byte offset, so it sees events regardless of which process produced them.
The journal is the single source of truth for cross-process live state.

Thread-safe: extraction runs in a ThreadPoolExecutor, so multiple workers publish
concurrently. Each subscriber gets its own queue. Journal appends are serialized
under the same lock; O_APPEND writes of one line are atomic on local filesystems.
"""
from __future__ import annotations

import json
import os
import queue
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path


@dataclass
class Event:
    kind: str                     # doc_status | doc_progress | unit | run_status | log | error | fallback | stage
    topic: str = ""
    seq: int = 0
    ts: float = 0.0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    def __init__(self, history_limit: int = 2000,
                 journal_path: "str | Path | None" = None) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: list[Event] = []
        self._history_limit = history_limit
        self._seq = 0
        # Optional append-only journal for cross-process observability. Events are
        # written as one JSON line each; readers tail by byte offset (read_journal).
        self._journal_path = Path(journal_path) if journal_path else None
        if self._journal_path:
            self._journal_path.parent.mkdir(parents=True, exist_ok=True)
            self._journal_path.touch(exist_ok=True)
            try:
                self._journal_path.chmod(0o600)
            except OSError:
                pass
        # Monotonic wall-clock source injected so the pipeline never calls
        # time.time() itself in a way that complicates testing; default real.
        self._clock = time.time

    def publish(self, kind: str, topic: str = "", **data) -> Event:
        with self._lock:
            self._seq += 1
            ev = Event(kind=kind, topic=topic, seq=self._seq,
                       ts=self._clock(), data=data)
            self._history.append(ev)
            if len(self._history) > self._history_limit:
                self._history = self._history[-self._history_limit:]
            if self._journal_path:
                self._append_journal(ev)
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev

    def _append_journal(self, ev: Event) -> None:
        """Append one event as a JSON line. Called under self._lock. A journal
        write must never crash a run, so I/O errors are swallowed (observability
        is best-effort; the run itself is the thing that matters)."""
        try:
            line = json.dumps(ev.to_dict(), ensure_ascii=False)
            with open(self._journal_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        except OSError:
            pass

    def subscribe(self) -> queue.Queue:
        q: queue.Queue = queue.Queue(maxsize=10000)
        with self._lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q: queue.Queue) -> None:
        with self._lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def history(self) -> list[Event]:
        with self._lock:
            return list(self._history)


def read_journal(path: "str | Path", offset: int = 0
                 ) -> "tuple[list[dict], int]":
    """Read event dicts appended to the journal since `offset` (a byte position).

    Returns (events, new_offset). A partial trailing line (a write in flight) is
    left unconsumed: new_offset points at its start so the next read picks it up
    once complete. Missing file → ([], offset). This is how the dashboard sees
    events produced by another process (e.g. a CLI run)."""
    p = Path(path)
    if not p.is_file():
        return [], offset
    try:
        size = p.stat().st_size
        if offset > size:      # file was truncated/rotated — restart from 0
            offset = 0
        with open(p, "r", encoding="utf-8") as f:
            f.seek(offset)
            chunk = f.read()
            consumed = f.tell()
    except OSError:
        return [], offset
    events: list[dict] = []
    # Only consume through the last complete newline; keep any partial tail.
    last_nl = chunk.rfind("\n")
    if last_nl == -1:
        return [], offset
    complete = chunk[:last_nl]
    new_offset = offset + len(complete.encode("utf-8")) + 1  # +1 for the newline
    # Use each line's byte position as its cross-process monotonic sequence.
    # Counting every preceding line on every 1s tail poll made a 16 MB journal get
    # reread in full once per second (quadratic over a long run). Byte positions
    # are already unique, ordered and available from the cursor at O(new data).
    line_offset = offset
    for line in complete.splitlines():
        encoded_size = len(line.encode("utf-8")) + 1
        line = line.strip()
        if not line:
            line_offset += encoded_size
            continue
        try:
            ev = json.loads(line)
        except (json.JSONDecodeError, ValueError):
            line_offset += encoded_size
            continue
        ev["seq"] = line_offset + 1
        events.append(ev)
        line_offset += encoded_size
    return events, new_offset
