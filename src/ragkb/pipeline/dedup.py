"""Dedup — deterministic near-duplicate collapse (no LLM).

Dozens of docs across overlapping topics produce QA units asking effectively the
same thing. We collapse them BEFORE export so the vector DB isn't polluted:
- Normalize the query (strip whitespace, unify 全角/半角 punctuation, casefold).
- Exact-hash on the normalized query → instant exact-dup merge.
- rapidfuzz token_set_ratio > threshold → near-dup merge.

When two units merge, we keep the longer/more-complete answer and UNION their
provenance + paraphrases (so we don't lose any query key or source). Semantic
dedup is left to the vector DB at query time — cheap deterministic dedup here is
enough at this scale.
"""
from __future__ import annotations

import re
import unicodedata

from rapidfuzz import fuzz

from ragkb.pipeline.units import QAUnit

_DEFAULT_THRESHOLD = 90.0


def _normalize(q: str) -> str:
    s = unicodedata.normalize("NFKC", q or "")     # 全角→半角, etc.
    s = re.sub(r"\s+", "", s)
    s = re.sub(r"[?？!！。，,、:：;；]+", "", s)
    return s.casefold()


def _merge(keep: QAUnit, drop: QAUnit) -> QAUnit:
    """Fold `drop` into `keep`: keep the longer answer, union paraphrases +
    sources + the dropped query as a paraphrase key (no key is lost)."""
    if len(drop.answer) > len(keep.answer):
        keep.answer = drop.answer
    existing = set(keep.query_keys())
    for k in [drop.query, *drop.paraphrases]:
        if k and k.strip() and k.strip() not in existing:
            keep.paraphrases.append(k.strip())
            existing.add(k.strip())
    keep.sources.extend(drop.sources)
    keep.needs_review = keep.needs_review or drop.needs_review
    return keep


def dedup_qa(units: list[QAUnit], threshold: float = _DEFAULT_THRESHOLD) -> list[QAUnit]:
    """Collapse exact + near-duplicate QA units. Order-stable on first occurrence."""
    kept: list[QAUnit] = []
    norms: list[str] = []
    exact: dict[str, int] = {}
    for u in units:
        nq = _normalize(u.query)
        if nq and nq in exact:
            _merge(kept[exact[nq]], u)
            continue
        # near-dup scan against already-kept norms
        hit = -1
        for idx, kn in enumerate(norms):
            if kn and fuzz.token_set_ratio(nq, kn) >= threshold:
                hit = idx
                break
        if hit >= 0:
            _merge(kept[hit], u)
            continue
        exact[nq] = len(kept)
        kept.append(u)
        norms.append(nq)
    return kept
