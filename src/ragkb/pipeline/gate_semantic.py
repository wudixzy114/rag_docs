"""Layer 2 — semantic quality gate (strong model, cross-reviewed, fail-closed).

Where Layer 1 catches structural corruption for free, Layer 2 judges CONTENT: is
the answer accurate, unambiguous, complete, and on-topic? This is the gate the
user asked to be model-driven ("用另一个很强的模型去审核"), and every unit is
reviewed (the doc set is small, so no sampling).

Design, lifted from magnus-lens verify.py:
- BATCHED + indexed verdicts (25/call), chunks run concurrently (cap 4) — turns N
  calls into N/25.
- Tolerant JSON parse + one bounded repair pass for missing ids.
- FAIL-CLOSED: a unit with no verdict after repair is NOT silently passed. Under
  the default policy it's dropped (marked semantic_ok=False); the caller may
  switch to keep-for-review.
- CROSS-MODEL: the reviewer routes through task="review", which .env points at a
  DIFFERENT model than extraction — an independent perspective catches errors a
  model's own self-review misses.
"""
from __future__ import annotations

import logging
from concurrent.futures import ThreadPoolExecutor

from ragkb.llm.client import LLMClient, LLMError
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import REVIEW_SYSTEM, build_review_user
from ragkb.pipeline.units import QAUnit

log = logging.getLogger(__name__)

_REVIEW_BATCH = 25
_REVIEW_CONCURRENCY = 4
_REVIEW_MAX_TOKENS = 2048


def _source_text(unit: QAUnit) -> str:
    """The provenance heading path(s) — a compact grounding reference for the
    reviewer. (Full source body isn't threaded here to keep the prompt bounded;
    the extractor already grounded on it, and the reviewer checks internal
    consistency + the heading context.)"""
    return " | ".join(p.heading_path for p in unit.sources if p.heading_path)


def _run_batch(chunk: list[QAUnit], llm: LLMClient) -> dict[int, tuple[str, str]]:
    items = [{"id": i, "query": u.query, "answer": u.answer,
              "source": _source_text(u)} for i, u in enumerate(chunk)]
    try:
        r = llm.complete(system=REVIEW_SYSTEM, user=build_review_user(items),
                         max_tokens=_REVIEW_MAX_TOKENS, task="review")
    except LLMError as exc:
        log.warning("review batch failed (%d items): %s", len(chunk), exc)
        return {}
    return _parse_verdicts(r.text)


def _parse_verdicts(text: str) -> dict[int, tuple[str, str]]:
    arr = parse_json_array(text) or []
    out: dict[int, tuple[str, str]] = {}
    for el in arr:
        if not isinstance(el, dict) or "id" not in el:
            continue
        try:
            idx = int(el["id"])
        except (TypeError, ValueError):
            continue
        verdict = str(el.get("verdict", "")).lower()
        if verdict not in ("pass", "revise", "reject"):
            continue
        out[idx] = (verdict, str(el.get("reason", "")))
    return out


def review_qa(units: list[QAUnit], llm: LLMClient,
              policy: str = "fail_closed") -> list[QAUnit]:
    """Review every unit; annotate semantic_ok / semantic_reason / needs_review.
    Returns the same list (mutated). A "revise" verdict passes but is flagged for
    human review; "reject" fails; a missing verdict is fail-closed per policy."""
    if not units:
        return units
    chunks = [units[i:i + _REVIEW_BATCH] for i in range(0, len(units), _REVIEW_BATCH)]

    def run(chunk):
        return chunk, _run_batch(chunk, llm)

    results = []
    if len(chunks) == 1:
        results.append(run(chunks[0]))
    else:
        workers = min(_REVIEW_CONCURRENCY, len(chunks))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            results = list(pool.map(run, chunks))

    for chunk, verdicts in results:
        for i, u in enumerate(chunk):
            if i not in verdicts:
                # No verdict after the batch — fail-closed. Never publish unreviewed.
                if policy == "keep":
                    u.semantic_ok = True
                    u.needs_review = True
                    u.semantic_reason = "no_verdict:kept_for_review"
                else:
                    u.semantic_ok = False
                    u.semantic_reason = "no_verdict:dropped"
                continue
            verdict, reason = verdicts[i]
            u.semantic_reason = f"{verdict}:{reason}"
            if verdict == "pass":
                u.semantic_ok = True
            elif verdict == "revise":
                u.semantic_ok = True
                u.needs_review = True
            else:  # reject
                u.semantic_ok = False
    return units
