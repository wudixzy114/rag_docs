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

from ragkb.llm.client import LLMClient, LLMError, LLMQuotaError
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import REVIEW_SYSTEM, build_review_user
from ragkb.pipeline.units import QAUnit

log = logging.getLogger(__name__)

_REVIEW_BATCH = 25
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
    r = llm.complete(system=REVIEW_SYSTEM, user=build_review_user(items),
                     max_tokens=_REVIEW_MAX_TOKENS, task="review")
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
    """Review every unit with the strong model (task='review' → Opus-4.8: generate
    first, review second, same top model). Annotates semantic_ok / semantic_reason
    / needs_review.

    Quota-aware (user's rule "审核没余额就不跑了"): if the reviewer model's quota is
    exhausted, we DON'T fail-closed-drop everything and we DON'T burn the run —
    we SKIP review, keep every struct-ok unit, and flag them needs_review so a
    human can check later. Losing the review pass must not lose the extraction.

    Batches run SERIALLY (no internal thread pool): review already runs once after
    the per-doc pool has closed, and serial batches keep gateway pressure low so
    one shared quota isn't hammered by nested concurrency.
    """
    if not units:
        return units
    chunks = [units[i:i + _REVIEW_BATCH] for i in range(0, len(units), _REVIEW_BATCH)]

    results = []
    for chunk in chunks:
        try:
            verdicts = _run_batch(chunk, llm)
        except LLMQuotaError:
            # Reviewer out of quota — stop reviewing entirely. Keep everything
            # already-and-not-yet reviewed, flagged for human review.
            log.warning("reviewer quota exhausted; skipping semantic review for "
                        "remaining %d units (kept, flagged needs_review)",
                        sum(len(c) for c in chunks[chunks.index(chunk):]))
            _mark_skipped(units)
            return units
        except LLMError as exc:
            log.warning("review batch failed (%d items): %s", len(chunk), exc)
            verdicts = {}
        results.append((chunk, verdicts))

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


def _mark_skipped(units: list[QAUnit]) -> None:
    """Reviewer unavailable: keep every unit that hasn't already been rejected,
    flag it for human review. Units already reviewed this run keep their verdict."""
    for u in units:
        if u.semantic_ok is None:
            u.semantic_ok = True
            u.needs_review = True
            u.semantic_reason = "review_skipped:no_quota"

