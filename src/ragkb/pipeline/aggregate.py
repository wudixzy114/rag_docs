"""Cross-file aggregation.

Information about one topic is scattered across sections and (potentially) docs.
After extraction, units are grouped by topic (= source folder, the natural
职责 boundary in this material) and consolidated:

- Within a topic, near-duplicate queries are already collapsed by dedup; here we
  additionally merge units whose ANSWERS are near-identical even if the queries
  differ (same fix asked two ways), unioning their query keys — aggressive merge,
  the user's choice.
- Provenance is unioned so a merged unit records every source it drew from
  (traceability under aggressive merge — the reason we keep provenance at all).

Kept deterministic (rapidfuzz on answers) to avoid burning tokens on a step that
mostly reshuffles; genuine semantic conflicts are surfaced by the Layer-2 review
verdicts already attached to each unit, and by the needs_review flag.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from ragkb.pipeline.units import QAUnit

_ANSWER_MERGE_THRESHOLD = 88.0


def _norm_answer(a: str) -> str:
    s = unicodedata.normalize("NFKC", a or "")
    s = re.sub(r"\s+", "", s)
    return s.casefold()


def _fold(keep: QAUnit, drop: QAUnit) -> None:
    existing = set(keep.query_keys())
    for k in drop.query_keys():
        if k not in existing:
            keep.paraphrases.append(k)
            existing.add(k)
    if len(drop.answer) > len(keep.answer):
        keep.answer = drop.answer
    keep.sources.extend(drop.sources)
    keep.needs_review = keep.needs_review or drop.needs_review


def aggregate_by_topic(units: list[QAUnit]) -> list[QAUnit]:
    """Merge units with near-identical answers (across the whole set, grouped by
    topic). Order-stable on first occurrence."""
    by_topic: dict[str, list[QAUnit]] = {}
    order: list[str] = []
    for u in units:
        topic = u.sources[0].topic if u.sources else ""
        if topic not in by_topic:
            by_topic[topic] = []
            order.append(topic)
        by_topic[topic].append(u)

    out: list[QAUnit] = []
    for topic in order:
        kept: list[QAUnit] = []
        norms: list[str] = []
        for u in by_topic[topic]:
            na = _norm_answer(u.answer)
            hit = -1
            for idx, kn in enumerate(norms):
                if kn and na and fuzz.ratio(na, kn) >= _ANSWER_MERGE_THRESHOLD:
                    hit = idx
                    break
            if hit >= 0:
                _fold(kept[hit], u)
            else:
                kept.append(u)
                norms.append(na)
        out.extend(kept)
    return out
