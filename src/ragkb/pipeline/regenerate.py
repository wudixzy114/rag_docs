"""One bounded, source-grounded regeneration pass for failed QA reviews."""
from __future__ import annotations

import logging
from collections.abc import Callable

from ragkb.llm.client import LLMClient, LLMError
from ragkb.pipeline.batching import pack_by_size
from ragkb.pipeline.gate_struct import gate_qa
from ragkb.pipeline.gate_semantic import review_qa
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import REGENERATE_SYSTEM, build_regenerate_user
from ragkb.pipeline.units import QAUnit

log = logging.getLogger(__name__)

StageCallback = Callable[[QAUnit, str, str, str], None]


def review_with_regeneration(units: list[QAUnit], llm: LLMClient, *,
                             max_attempts: int = 1,
                             reviewer_model: str | None = None,
                             on_stage: StageCallback | None = None) -> list[QAUnit]:
    """Review, regenerate failed units at most N times, then review each retry."""
    if not units:
        return units
    for unit in units:
        if on_stage:
            on_stage(unit, "review", "running", "initial review")
    review_qa(units, llm, model=reviewer_model)

    for _ in range(max_attempts):
        retryable = [u for u in units if not u.semantic_ok and u.review_attempts < max_attempts]
        if not retryable:
            break
        for unit in retryable:
            if on_stage:
                on_stage(unit, "regenerate", "running", unit.semantic_reason)
        regenerate_failed_qa(retryable, llm)
        regenerated = [u for u in retryable if u.semantic_ok is None]
        for unit in retryable:
            if on_stage:
                status = "done" if unit in regenerated else "failed"
                on_stage(unit, "regenerate", status, unit.semantic_reason)
        if regenerated:
            for unit in regenerated:
                if on_stage:
                    on_stage(unit, "review", "running", "review regenerated answer")
            review_qa(regenerated, llm, model=reviewer_model)

    for unit in units:
        if unit.semantic_ok:
            unit.publication_status = "approved"
            unit.needs_review = False
            status = "done"
        else:
            unit.publication_status = "failed_review"
            unit.needs_review = True
            status = "failed"
        if on_stage:
            on_stage(unit, "review", status, unit.semantic_reason)
    return units


def regenerate_failed_qa(units: list[QAUnit], llm: LLMClient,
                         model: str | None = None) -> list[QAUnit]:
    """Regenerate each failed unit once, preserving provenance and failure history."""
    indexed = list(enumerate(units))
    batches = pack_by_size(
        indexed,
        lambda pair: len(pair[1].answer) + sum(len(p.source_excerpt) for p in pair[1].sources),
        max_items=8, max_chars=28000)
    for batch in batches:
        items = []
        for idx, unit in batch:
            evidence = "\n\n".join(
                f"[{p.heading_path or p.doc_title}]\n{p.source_excerpt}"
                for p in unit.sources)[:12000]
            items.append({
                "id": idx, "previous_query": unit.query,
                "previous_answer": unit.answer,
                "review_failure": unit.semantic_reason,
                "source": evidence,
            })
        try:
            result = llm.complete(
                system=REGENERATE_SYSTEM, user=build_regenerate_user(items),
                max_tokens=6144, task="extract", model=model)
            rows = parse_json_array(result.text)
            if rows is None:
                raise LLMError("regeneration returned invalid JSON")
        except LLMError as exc:
            log.warning("QA regeneration batch failed (%d items): %s", len(batch), exc)
            rows = []
        got = {}
        for row in rows:
            if not isinstance(row, dict):
                continue
            try:
                idx = int(row.get("id"))
            except (TypeError, ValueError):
                continue
            got[idx] = row
        for idx, unit in batch:
            unit.review_history.append(unit.semantic_reason or "review_failed")
            unit.review_attempts += 1
            row = got.get(idx)
            if not row:
                unit.semantic_ok = False
                unit.needs_review = True
                unit.publication_status = "failed_review"
                unit.semantic_reason = "regeneration_failed:no_result"
                continue
            unit.query = str(row.get("query", "")).strip()
            unit.answer = str(row.get("answer", "")).strip()
            unit.paraphrases = []
            gate_qa(unit)
            if not unit.struct_ok:
                unit.semantic_ok = False
                unit.needs_review = True
                unit.publication_status = "failed_review"
                unit.semantic_reason = f"regeneration_failed:{unit.struct_reason}"
                continue
            unit.semantic_ok = None
            unit.needs_review = False
            unit.publication_status = "pending"
            unit.semantic_reason = ""
    return units
