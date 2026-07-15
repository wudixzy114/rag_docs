"""Paraphrase generation — the recall booster.

For each QA unit, generate several user-phrased variants of the query, all
pointing at the SAME answer. In the vector DB every variant becomes its own
key row, widening the match surface (the user's explicit ask: 增加键的匹配数量).
Emphasis on symptom-style phrasings ("内存爆了" for an OOM answer) to bridge the
user话术 vs 文档话术 gap.

Cached per (query + PARAPHRASE_VERSION) so re-runs are free.
"""
from __future__ import annotations

import logging

from ragkb.llm.client import LLMClient, LLMError
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import (PARAPHRASE_VERSION, build_paraphrase_user,
                                    build_batch_paraphrase_user, PARAPHRASE_SYSTEM)
from ragkb.pipeline.batching import pack_by_size
from ragkb.pipeline.units import QAUnit
from ragkb.store.cache import Cache, key_for

log = logging.getLogger(__name__)


def add_paraphrases(unit: QAUnit, llm: LLMClient, cache: Cache,
                    n: int = 4) -> QAUnit:
    """Fill unit.paraphrases with up to n user-phrased query variants."""
    q = unit.query.strip()
    if not q:
        return unit
    ck = key_for(q, str(n), PARAPHRASE_VERSION)
    cached = cache.get("paraphrase", ck)
    if cached is not None:
        unit.paraphrases = cached
        return unit
    try:
        r = llm.complete(system=PARAPHRASE_SYSTEM,
                         user=build_paraphrase_user(q, unit.answer, n),
                         max_tokens=1024, task="paraphrase",
                         chain=llm.settings.simple_chain())
        arr = parse_json_array(r.text) or []
    except LLMError as exc:
        log.warning("paraphrase failed for %r: %s", q[:40], exc)
        arr = []
    variants = []
    for el in arr:
        s = str(el).strip()
        if s and s != q:
            variants.append(s)
    unit.paraphrases = variants
    cache.put("paraphrase", ck, variants)
    return unit


def add_paraphrases_batch(units: list[QAUnit], llm: LLMClient, cache: Cache,
                          n: int = 4) -> list[QAUnit]:
    """Generate query variants in cache-aware Flash batches."""
    pending: list[tuple[int, QAUnit, str]] = []
    for idx, unit in enumerate(units):
        q = unit.query.strip()
        if not q:
            continue
        ck = key_for(q, str(n), PARAPHRASE_VERSION)
        cached = cache.get("paraphrase", ck)
        if cached is not None:
            unit.paraphrases = list(cached)
        else:
            pending.append((idx, unit, ck))

    for batch in pack_by_size(
            pending, lambda x: len(x[1].query) + min(len(x[1].answer), 800),
            max_items=20, max_chars=14000):
        payload = [{"id": idx, "query": unit.query, "answer": unit.answer[:800]}
                   for idx, unit, _ in batch]
        got: dict[int, list[str]] = {}
        try:
            result = llm.complete(
                system=PARAPHRASE_SYSTEM,
                user=build_batch_paraphrase_user(payload, n), max_tokens=2048,
                task="paraphrase", chain=llm.settings.simple_chain())
            for row in parse_json_array(result.text) or []:
                if not isinstance(row, dict):
                    continue
                try:
                    row_id = int(row.get("id"))
                except (TypeError, ValueError):
                    continue
                values = row.get("variants", [])
                if isinstance(values, list):
                    got[row_id] = [str(v).strip() for v in values if str(v).strip()]
        except LLMError as exc:
            log.warning("paraphrase batch failed (%d units): %s", len(batch), exc)
        for idx, unit, ck in batch:
            variants, seen = [], {unit.query.strip()}
            for value in got.get(idx, []):
                if value not in seen and len(variants) < n:
                    seen.add(value)
                    variants.append(value)
            unit.paraphrases = variants
            if idx in got:  # missing rows are retried on a later run
                cache.put("paraphrase", ck, variants)
    return units
