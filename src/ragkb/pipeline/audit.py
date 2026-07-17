"""Durable stage snapshots and text diffs for pipeline traceability."""
from __future__ import annotations

import difflib
import hashlib
import json
import threading
import time
from pathlib import Path


def _sha(text: str) -> str:
    return hashlib.sha256((text or "").encode("utf-8")).hexdigest()


class AuditStore:
    """One atomic audit document per topic; every stage keeps all attempts."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self._lock = threading.RLock()

    @staticmethod
    def _name(topic: str) -> str:
        return hashlib.sha256(topic.encode("utf-8")).hexdigest()[:24] + ".json"

    def _path(self, topic: str) -> Path:
        return self.root / self._name(topic)

    def _load(self, topic: str) -> dict:
        path = self._path(topic)
        if not path.is_file():
            return {"topic": topic, "stages": {}}
        try:
            data = json.loads(path.read_text("utf-8"))
        except (OSError, ValueError, json.JSONDecodeError):
            return {"topic": topic, "stages": {}}
        return data if data.get("topic") == topic else {"topic": topic, "stages": {}}

    def record(self, topic: str, stage: str, *, status: str,
               input_text: str = "", output_text: str = "",
               metadata: dict | None = None, error: str = "") -> dict:
        before, after = input_text or "", output_text or ""
        diff = "".join(difflib.unified_diff(
            before.splitlines(keepends=True), after.splitlines(keepends=True),
            fromfile=f"{stage}:input", tofile=f"{stage}:output", n=3))
        record = {
            "attempt": 0, "ts": time.time(), "status": status,
            "input_sha": _sha(before), "output_sha": _sha(after),
            "input_chars": len(before), "output_chars": len(after),
            "input_text": before, "output_text": after, "diff": diff,
            "metadata": metadata or {}, "error": error,
        }
        with self._lock:
            data = self._load(topic)
            attempts = data.setdefault("stages", {}).setdefault(stage, [])
            record["attempt"] = len(attempts) + 1
            attempts.append(record)
            self.root.mkdir(parents=True, exist_ok=True)
            path = self._path(topic)
            tmp = path.with_suffix(".tmp")
            tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2), "utf-8")
            tmp.replace(path)
            try:
                self.root.chmod(0o700)
                path.chmod(0o600)
            except OSError:
                pass
        return record

    def get(self, topic: str) -> dict:
        with self._lock:
            return self._load(topic)

    def summary(self, topic: str) -> dict:
        data = self.get(topic)
        out = {}
        for stage, attempts in data.get("stages", {}).items():
            if not attempts:
                continue
            last = attempts[-1]
            out[stage] = {key: last.get(key) for key in (
                "attempt", "ts", "status", "input_sha", "output_sha",
                "input_chars", "output_chars", "error")}
        return out
