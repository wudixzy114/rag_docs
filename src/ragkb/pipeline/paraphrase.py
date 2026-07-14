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
from ragkb.pipeline.prompts import PARAPHRASE_VERSION, build_paraphrase_user, PARAPHRASE_SYSTEM
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
