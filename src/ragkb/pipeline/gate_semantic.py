"""Layer 2 — QA and SOP semantic quality gates (strong model, fail-closed).

Where Layer 1 catches structural corruption for free, Layer 2 judges CONTENT:
is the QA answer or SOP accurate, unambiguous, complete, and on-topic? Every
unit is reviewed against its source evidence; there is no sampling.

Design:
- Size-bounded indexed batches turn N calls into a small number of calls without
  allowing one huge evidence item to overflow the context budget.
- Tolerant JSON parse + one bounded repair pass for missing ids.
- FAIL-CLOSED: a unit with no verdict after repair is NOT silently passed. Under
  the default policy it's dropped (marked semantic_ok=False); the caller may
  switch to keep-for-review.
- Evidence-grounded: the reviewer receives the masked source excerpt and image
  transcript, plus every generated query variant. A separate reviewer model can
  still be configured, while the default uses Gemini 3.1 Pro for quality.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from ragkb.llm.client import LLMClient, LLMError, LLMQuotaError
from ragkb.pipeline.batching import pack_by_size
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import (
    REVIEW_SYSTEM,
    SOP_REVIEW_SYSTEM,
    build_review_user,
    build_sop_review_user,
)
from ragkb.pipeline.units import QAUnit, SOPUnit

log = logging.getLogger(__name__)

_REVIEW_BATCH = 12
_REVIEW_MAX_CHARS = 36000
_REVIEW_MAX_TOKENS = 4096


@dataclass
class _Verdict:
    verdict: str
    reason: str
    valid_paraphrases: list[str] | None = None


def _source_text(unit: QAUnit | SOPUnit, max_chars: int = 12000) -> str:
    """Grounding evidence, including the source excerpt rather than headings only."""
    parts = []
    for p in unit.sources:
        excerpt = p.source_excerpt.strip()
        parts.append(f"[{p.heading_path or p.doc_title}]\n{excerpt}" if excerpt
                     else f"[{p.heading_path or p.doc_title}]（原文缺失）")
    return "\n\n".join(parts)[:max_chars]


def _run_batch(indexed: list[tuple[int, QAUnit]], llm: LLMClient,
               model: str | None = None) -> dict[int, _Verdict]:
    items = [{"id": i, "query": u.query, "answer": u.answer,
              "paraphrases": u.paraphrases,
              "source": _source_text(u)} for i, u in indexed]
    r = llm.complete(system=REVIEW_SYSTEM, user=build_review_user(items),
                     max_tokens=_REVIEW_MAX_TOKENS, task="review", model=model)
    return _parse_verdicts(r.text)


def _parse_verdicts(text: str) -> dict[int, _Verdict]:
    arr = parse_json_array(text) or []
    out: dict[int, _Verdict] = {}
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
        valid = el.get("valid_paraphrases")
        valid = [str(v).strip() for v in valid if str(v).strip()] if isinstance(valid, list) else None
        out[idx] = _Verdict(
            verdict=verdict, reason=str(el.get("reason", "")),
            valid_paraphrases=valid)
    return out


def review_qa(units: list[QAUnit], llm: LLMClient,
              policy: str = "fail_closed", model: str | None = None) -> list[QAUnit]:
    """Review every unit with the strong task route (Gemini 3.1 Pro by default).
    Annotates semantic_ok / semantic_reason / needs_review.

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
    indexed = list(enumerate(units))
    chunks = pack_by_size(
        indexed,
        lambda pair: len(pair[1].query) + len(pair[1].answer) + len(_source_text(pair[1])),
        max_items=_REVIEW_BATCH, max_chars=_REVIEW_MAX_CHARS)

    verdicts: dict[int, _Verdict] = {}
    for chunk_no, chunk in enumerate(chunks):
        try:
            batch_verdicts = _run_batch(chunk, llm, model=model)
        except LLMQuotaError:
            # Reviewer out of quota — stop reviewing entirely. Keep everything
            # already-and-not-yet reviewed, flagged for human review.
            log.warning("reviewer quota exhausted; skipping semantic review for "
                        "remaining %d units (kept, flagged needs_review)",
                        sum(len(c) for c in chunks[chunk_no:]))
            _apply_known(units, verdicts)
            _mark_skipped(units)
            return units
        except LLMError as exc:
            log.warning("review batch failed (%d items): %s", len(chunk), exc)
            batch_verdicts = {}
        verdicts.update(batch_verdicts)

        # One bounded repair call containing only omitted IDs. This makes the
        # documented repair behavior real and avoids paying to re-review valid rows.
        missing = [pair for pair in chunk if pair[0] not in batch_verdicts]
        if missing:
            try:
                verdicts.update(_run_batch(missing, llm, model=model))
            except LLMQuotaError:
                _apply_known(units, verdicts)
                _mark_skipped(units)
                return units
            except LLMError as exc:
                log.warning("review repair failed (%d items): %s", len(missing), exc)

    for i, u in indexed:
        if i not in verdicts:
            u.semantic_ok = False
            u.needs_review = True
            u.publication_status = "failed_review"
            u.semantic_reason = "no_verdict"
            continue
        result = verdicts[i]
        u.semantic_reason = f"{result.verdict}:{result.reason}"
        _filter_paraphrases(u, result.valid_paraphrases)
        if result.verdict == "pass":
            u.semantic_ok = True
            u.publication_status = "approved"
        elif result.verdict == "revise":
            u.semantic_ok = False
            u.needs_review = True
            u.publication_status = "failed_review"
        else:
            u.semantic_ok = False
            u.needs_review = True
            u.publication_status = "failed_review"
    return units


def _apply_known(units: list[QAUnit], verdicts: dict[int, _Verdict]) -> None:
    """Preserve completed batch verdicts if reviewer quota ends mid-run."""
    for idx, result in verdicts.items():
        if not 0 <= idx < len(units):
            continue
        unit = units[idx]
        unit.semantic_reason = f"{result.verdict}:{result.reason}"
        _filter_paraphrases(unit, result.valid_paraphrases)
        if result.verdict == "pass":
            unit.semantic_ok = True
            unit.publication_status = "approved"
        else:
            unit.semantic_ok = False
            unit.needs_review = True
            unit.publication_status = "failed_review"


def _filter_paraphrases(unit: QAUnit, approved: list[str] | None) -> None:
    """Keep only exact submitted candidates; the reviewer cannot invent new keys."""
    allowed = set(approved or [])
    unit.paraphrases = [value for value in unit.paraphrases if value in allowed]


def _mark_skipped(units: list[QAUnit]) -> None:
    """Reviewer unavailable: retain but never approve an unreviewed unit."""
    for u in units:
        if u.semantic_ok is None:
            u.paraphrases = []
            u.semantic_ok = False
            u.needs_review = True
            u.publication_status = "failed_review"
            u.semantic_reason = "review_unavailable:no_quota"


# ------------------------------------------------------------------- SOP ----
_SOP_REVIEW_BATCH = 4
_SOP_REVIEW_MAX_CHARS = 52000
_SOP_REVIEW_MAX_TOKENS = 4096


@dataclass
class _SOPVerdict:
    verdict: str
    reason: str
    valid_entry_questions: list[str] | None = None


def _run_sop_batch(indexed: list[tuple[int, SOPUnit]], llm: LLMClient,
                   model: str | None = None) -> dict[int, _SOPVerdict]:
    items = [{"id": i, "title": u.title, "markdown": u.markdown,
              "entry_questions": u.entry_questions,
              "source": _source_text(u, max_chars=16000)} for i, u in indexed]
    result = llm.complete(
        system=SOP_REVIEW_SYSTEM, user=build_sop_review_user(items),
        max_tokens=_SOP_REVIEW_MAX_TOKENS, task="review", model=model)
    return _parse_sop_verdicts(result.text)


def _parse_sop_verdicts(text: str) -> dict[int, _SOPVerdict]:
    arr = parse_json_array(text) or []
    out: dict[int, _SOPVerdict] = {}
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
        valid = el.get("valid_entry_questions")
        valid = ([str(v).strip() for v in valid if str(v).strip()]
                 if isinstance(valid, list) else None)
        out[idx] = _SOPVerdict(
            verdict=verdict, reason=str(el.get("reason", "")),
            valid_entry_questions=valid)
    return out


def review_sop(units: list[SOPUnit], llm: LLMClient,
               model: str | None = None) -> list[SOPUnit]:
    """Review every SOP against its source evidence, fail-closed.

    Missing/invalid verdicts and reviewer errors cannot approve an SOP. A quota
    failure preserves the candidate in results.json for manual recovery but marks
    it failed_review so neither export path can publish it.
    """
    if not units:
        return units
    indexed = list(enumerate(units))
    chunks = pack_by_size(
        indexed,
        lambda pair: (len(pair[1].title) + len(pair[1].markdown)
                      + len(_source_text(pair[1], max_chars=16000))),
        max_items=_SOP_REVIEW_BATCH, max_chars=_SOP_REVIEW_MAX_CHARS)

    verdicts: dict[int, _SOPVerdict] = {}
    for chunk_no, chunk in enumerate(chunks):
        try:
            batch_verdicts = _run_sop_batch(chunk, llm, model=model)
        except LLMQuotaError:
            log.warning("reviewer quota exhausted; skipping SOP semantic review for "
                        "remaining %d units", sum(len(c) for c in chunks[chunk_no:]))
            _apply_known_sop(units, verdicts)
            _mark_skipped_sop(units)
            return units
        except LLMError as exc:
            log.warning("SOP review batch failed (%d items): %s", len(chunk), exc)
            batch_verdicts = {}
        verdicts.update(batch_verdicts)

        missing = [pair for pair in chunk if pair[0] not in batch_verdicts]
        if missing:
            try:
                verdicts.update(_run_sop_batch(missing, llm, model=model))
            except LLMQuotaError:
                _apply_known_sop(units, verdicts)
                _mark_skipped_sop(units)
                return units
            except LLMError as exc:
                log.warning("SOP review repair failed (%d items): %s", len(missing), exc)

    for i, unit in indexed:
        result = verdicts.get(i)
        if result is None:
            unit.semantic_ok = False
            unit.needs_review = True
            unit.publication_status = "failed_review"
            unit.semantic_reason = "no_verdict"
            continue
        _apply_sop_verdict(unit, result)
    return units


def _apply_sop_verdict(unit: SOPUnit, result: _SOPVerdict) -> None:
    unit.semantic_reason = f"{result.verdict}:{result.reason}"
    allowed = set(result.valid_entry_questions or [])
    unit.entry_questions = [q for q in unit.entry_questions if q in allowed]
    if result.verdict == "pass" and unit.entry_questions:
        unit.semantic_ok = True
        unit.needs_review = False
        unit.publication_status = "approved"
    else:
        unit.semantic_ok = False
        unit.needs_review = True
        unit.publication_status = "failed_review"
        if result.verdict == "pass":
            unit.semantic_reason = "revise:no_valid_entry_questions"


def _apply_known_sop(units: list[SOPUnit],
                     verdicts: dict[int, _SOPVerdict]) -> None:
    for idx, result in verdicts.items():
        if 0 <= idx < len(units):
            _apply_sop_verdict(units[idx], result)


def _mark_skipped_sop(units: list[SOPUnit]) -> None:
    for unit in units:
        if unit.semantic_ok is None:
            unit.semantic_ok = False
            unit.needs_review = True
            unit.publication_status = "failed_review"
            unit.semantic_reason = "review_unavailable:no_quota"
