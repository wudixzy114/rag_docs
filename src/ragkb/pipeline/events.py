"""In-process event bus for live observability.

The orchestrator publishes progress events (doc started, section extracted, unit
gated, doc done, error); the dashboard's SSE endpoint subscribes and streams them
to the browser. Also retains a bounded history so a late-connecting browser can
replay recent state, and the CLI can print a running log.

Thread-safe: extraction runs in a ThreadPoolExecutor, so multiple workers publish
concurrently. Each subscriber gets its own queue.
"""
from __future__ import annotations

import queue
import threading
import time
from dataclasses import dataclass, field, asdict


@dataclass
class Event:
    kind: str                     # doc_status | doc_progress | unit | run_status | log | error
    topic: str = ""
    seq: int = 0
    ts: float = 0.0
    data: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


class EventBus:
    def __init__(self, history_limit: int = 2000) -> None:
        self._lock = threading.Lock()
        self._subscribers: list[queue.Queue] = []
        self._history: list[Event] = []
        self._history_limit = history_limit
        self._seq = 0
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
            subs = list(self._subscribers)
        for q in subs:
            try:
                q.put_nowait(ev)
            except queue.Full:
                pass
        return ev

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
