"""Persistent human-in-the-loop decisions for exceptional pipeline cases."""
from __future__ import annotations

import hashlib
import json
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

from ragkb.parse.source import load_source


class HumanReviewRequired(RuntimeError):
    """Automation stopped at a deterministic decision boundary."""

    def __init__(self, message: str, decision_id: str) -> None:
        super().__init__(message)
        self.decision_id = decision_id


@dataclass
class Decision:
    decision_id: str
    topic: str
    stage: str
    status: str = "pending"          # pending | resolved
    recommended_action: str = "exclude"
    selected_action: str = ""         # include | exclude | retry | accept
    reason: str = ""
    evidence: list[str] = field(default_factory=list)
    options: list[str] = field(default_factory=lambda: ["include", "exclude"])
    created_at: float = 0.0
    resolved_at: float = 0.0

    def to_dict(self) -> dict:
        return asdict(self)


class DecisionStore:
    """Atomic JSON-backed decision queue shared by CLI and dashboard."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self._lock = threading.RLock()
        self._items: dict[str, Decision] = {}
        self.reload()

    @staticmethod
    def key(topic: str, stage: str) -> str:
        digest = hashlib.sha256(f"{stage}\0{topic}".encode("utf-8")).hexdigest()[:20]
        return f"{stage}:{digest}"

    def reload(self) -> None:
        with self._lock:
            if not self.path.is_file():
                return
            try:
                data = json.loads(self.path.read_text("utf-8"))
            except (OSError, ValueError, json.JSONDecodeError):
                return
            loaded = {}
            for raw in data.get("decisions", []):
                try:
                    item = Decision(**raw)
                except (TypeError, ValueError):
                    continue
                loaded[item.decision_id] = item
            self._items = loaded

    def _save(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {"decisions": [item.to_dict() for item in self._items.values()]}
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(self.path)
        try:
            self.path.chmod(0o600)
        except OSError:
            pass

    def propose(self, topic: str, stage: str, *, reason: str,
                evidence: list[str], recommended_action: str = "exclude",
                options: list[str] | None = None) -> Decision:
        decision_id = self.key(topic, stage)
        with self._lock:
            existing = self._items.get(decision_id)
            if existing:
                changed = existing.reason != reason or existing.evidence != list(evidence)
                # A materially new anomaly requires a new decision. Stable
                # preflight evidence keeps the operator's prior include/exclude.
                if changed and stage != "preflight" and existing.status == "resolved":
                    existing.status = "pending"
                    existing.selected_action = ""
                    existing.created_at = time.time()
                    existing.resolved_at = 0.0
                existing.reason = reason
                existing.evidence = list(evidence)
                self._save()
                return existing
            item = Decision(
                decision_id=decision_id, topic=topic, stage=stage,
                reason=reason, evidence=list(evidence),
                recommended_action=recommended_action,
                options=options or ["include", "exclude"], created_at=time.time())
            self._items[decision_id] = item
            self._save()
            return item

    def resolve(self, decision_id: str, action: str) -> Decision:
        with self._lock:
            item = self._items.get(decision_id)
            if item is None:
                raise KeyError(decision_id)
            if action not in item.options:
                raise ValueError(f"invalid action {action!r}; expected one of {item.options}")
            item.status = "resolved"
            item.selected_action = action
            item.resolved_at = time.time()
            self._save()
            return item

    def get_for(self, topic: str, stage: str) -> Decision | None:
        with self._lock:
            return self._items.get(self.key(topic, stage))

    def get(self, decision_id: str) -> Decision | None:
        with self._lock:
            return self._items.get(decision_id)

    def remove(self, decision_id: str) -> None:
        with self._lock:
            if self._items.pop(decision_id, None) is not None:
                self._save()

    def close_topic(self, topic: str, action: str, *, except_id: str = "") -> None:
        """Close obsolete pending decisions after a whole-topic exclusion."""
        with self._lock:
            changed = False
            for item in self._items.values():
                if (item.topic != topic or item.decision_id == except_id
                        or item.status != "pending"):
                    continue
                item.status = "resolved"
                item.selected_action = action
                item.resolved_at = time.time()
                changed = True
            if changed:
                self._save()

    def all(self, *, status: str | None = None) -> list[Decision]:
        with self._lock:
            values = list(self._items.values())
        if status:
            values = [item for item in values if item.status == status]
        return sorted(values, key=lambda item: (item.status != "pending", item.created_at))


_PATH_FLAGS = re.compile(
    r"(?:^|[/_\-（(【\[])(已私密|私密文档|机密文档|仅协作人访问|协作人可见|内部受限|"
    r"已废弃|废弃文档|已下线|停止维护)"
    r"(?:$|[/_\-）)】\]])", re.I)
_HEADER_FLAGS = [
    re.compile(r"^\s*#{0,3}\s*[【\[]\s*(已私密|私密|机密|仅协作人访问|协作人可见|内部受限|已废弃|废弃|已下线)\s*[】\]]", re.I),
    re.compile(r"^\s*(?:status|状态|文档状态)\s*[:：]\s*(已废弃|废弃|已下线|停止维护)\s*$", re.I),
    re.compile(r"^\s*(?:visibility|可见性)\s*[:：]\s*(private|私密|机密)\s*$", re.I),
    re.compile(r"^\s*(?:deprecated|废弃)\s*[:：]\s*(?:true|yes|是)\s*$", re.I),
]


def inspect_preflight(path: Path) -> tuple[list[str], list[str]]:
    """Return strong document-level flags and their exact evidence.

    The rules intentionally avoid matching ordinary body text such as "某 API 已废弃";
    only path labels and document metadata/header markers can quarantine a whole file.
    """
    flags: list[str] = []
    evidence: list[str] = []
    relative_hint = path.as_posix()
    for match in _PATH_FLAGS.finditer(relative_hint):
        value = match.group(1)
        flags.append("private" if ("私密" in value or "机密" in value
                                   or "协作人" in value or "受限" in value)
                     else "deprecated")
        evidence.append(f"路径标记：{value}")
    try:
        preview = load_source(path)[:12000]
    except (OSError, ValueError):
        preview = ""
    for line in preview.splitlines()[:80]:
        for pattern in _HEADER_FLAGS:
            match = pattern.search(line)
            if not match:
                continue
            value = match.group(1)
            flag = ("private" if (value.lower() == "private" or "私密" in value
                                   or "机密" in value or "协作人" in value
                                   or "受限" in value) else "deprecated")
            flags.append(flag)
            evidence.append(f"文档标记：{line.strip()[:240]}")
            break
    return list(dict.fromkeys(flags)), list(dict.fromkeys(evidence))
