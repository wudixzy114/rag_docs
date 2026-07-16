"""Extraction: a classified section → QA units or a SOP unit.

Both paths fuse the section body with its images' VISION transcriptions (not the
weak inline OCR) and detect truncation via finish_reason. Extraction returns raw
units; the deterministic gate (gate_struct) and semantic gate (gate_semantic)
run afterward. Provenance is attached here, at the point we still know the exact
source section.
"""
from __future__ import annotations

import logging

from ragkb.llm.client import LLMClient, LLMError
from ragkb.parse.model import Document, Section
from ragkb.pipeline.jsonutil import parse_json_array, parse_json_object
from ragkb.pipeline.prompts import (
    EXTRACT_SYSTEM, SOP_SYSTEM, EXTRACT_VERSION, SOP_VERSION,
    build_extract_user, build_batch_extract_user, build_sop_user, build_images_block,
)
from ragkb.pipeline.batching import pack_by_size
from ragkb.pipeline.scrub import mask
from ragkb.pipeline.units import Provenance, QAUnit, SOPUnit
from ragkb.store.cache import key_for

log = logging.getLogger(__name__)

_EXTRACT_MAX_TOKENS = 4096
_SOP_MAX_TOKENS = 6144
# A dense SOP section (long body + many image transcripts) can overflow the base
# budget and come back truncated → invalid JSON. Retry once at this larger budget
# before failing, so a big section isn't lost the way V2.5.0「亮点功能」was.
_SOP_RETRY_MAX_TOKENS = 12288
_BATCH_MAX_ITEMS = 8
_BATCH_MAX_CHARS = 18000


def _section_images(sec: Section) -> list[dict]:
    """Vision transcripts for the images in this section (own images only).
    Scrubbed pre-send so gateway content-safety doesn't 400 on MAC/IP in a chart."""
    out = []
    for im in sec.images:
        out.append({"rel_path": im.rel_path,
                    "transcript": mask(im.vision_text or im.inline_ocr),
                    "meaning": ""})
    return out


def _provenance(doc: Document, sec: Section) -> Provenance:
    evidence = mask(sec.body)
    image_text = "\n\n".join(
        f"[图片 {im.rel_path}]\n{mask(im.vision_text or im.inline_ocr)}"
        for im in sec.images if (im.vision_text or im.inline_ocr).strip())
    if image_text:
        evidence = f"{evidence}\n\n{image_text}".strip()
    return Provenance(
        topic=doc.topic, doc_title=doc.title,
        heading_path=sec.full_title, section_sid=sec.sid,
        source_sha=doc.source_sha,
        image_refs=[im.rel_path for im in sec.images],
        source_excerpt=evidence[:16000],
    )


def extract_qa(doc: Document, sec: Section, llm: LLMClient,
               model: str | None = None, cache=None,
               max_tokens: int = _EXTRACT_MAX_TOKENS) -> list[QAUnit]:
    """Extract QA pairs from a section. Truncation (finish_reason==length) is
    recorded on every unit so the struct gate can fail them and the orchestrator
    can retry with a larger budget. Cached by (section content + prompt version)
    so a re-run skips already-extracted sections instantly."""
    images = _section_images(sec)
    user = build_extract_user(sec.full_title, sec.title, mask(sec.body),
                              build_images_block(images))
    prov = _provenance(doc, sec)
    ck = None
    if cache is not None:
        ck = key_for(_content_sha(user), model or "task:extract", EXTRACT_VERSION)
        hit = cache.get("extract", ck)
        if hit is not None:
            return [_qa_from_dict(d, prov) for d in hit]
    try:
        r = llm.complete(system=EXTRACT_SYSTEM, user=user,
                         max_tokens=max_tokens, task="extract", model=model)
    except LLMError as exc:
        raise LLMError(f"extract_qa failed [{doc.topic} {sec.sid}]: {exc}") from exc
    truncated = r.finish_reason == "length"
    arr = parse_json_array(r.text)
    if arr is None:
        raise LLMError(f"extract_qa returned invalid JSON [{doc.topic} {sec.sid}]")
    units: list[QAUnit] = []
    raw: list[dict] = []
    for el in arr:
        if not isinstance(el, dict):
            continue
        q, a = str(el.get("query", "")).strip(), str(el.get("answer", "")).strip()
        if not q and not a:
            continue
        raw.append({"query": q, "answer": a, "truncated": truncated})
        units.append(QAUnit(query=q, answer=a, sources=[prov], truncated=truncated))
    # Only cache a clean (non-truncated) extraction — a truncated one must re-run.
    if cache is not None and ck and not truncated:
        cache.put("extract", ck, raw)
    return units


def extract_qa_sections(doc: Document, sections: list[Section], llm: LLMClient,
                        cache=None) -> list[QAUnit]:
    """Cache-aware, size-bounded extraction for multiple independent sections."""
    output: list[QAUnit] = []
    pending: list[tuple[Section, str, str | None]] = []
    for sec in sections:
        user = build_extract_user(sec.full_title, sec.title, mask(sec.body),
                                  build_images_block(_section_images(sec)))
        ck = key_for(_content_sha(user), "task:extract", EXTRACT_VERSION) if cache else None
        hit = cache.get("extract", ck) if cache and ck else None
        if hit is not None:
            prov = _provenance(doc, sec)
            output.extend(_qa_from_dict(d, prov) for d in hit)
        else:
            pending.append((sec, user, ck))

    for batch in pack_by_size(pending, lambda item: len(item[1]),
                              max_items=_BATCH_MAX_ITEMS, max_chars=_BATCH_MAX_CHARS):
        payload = [{"id": sec.sid, "heading_path": sec.full_title, "content": user}
                   for sec, user, _ in batch]
        parsed: dict[str, list[dict]] = {}
        truncated = False
        try:
            result = llm.complete(system=EXTRACT_SYSTEM,
                                  user=build_batch_extract_user(payload),
                                  max_tokens=_EXTRACT_MAX_TOKENS, task="extract")
            truncated = result.finish_reason == "length"
            for row in parse_json_array(result.text) or []:
                if not isinstance(row, dict) or "id" not in row:
                    continue
                items = row.get("items", [])
                if isinstance(items, list):
                    parsed[str(row["id"])] = [x for x in items if isinstance(x, dict)]
        except LLMError as exc:
            log.warning("batch extraction failed [%s, %d sections]: %s",
                        doc.topic, len(batch), exc)

        for sec, _user, ck in batch:
            rows = parsed.get(sec.sid)
            if rows is None or truncated:
                output.extend(extract_qa(
                    doc, sec, llm, cache=cache,
                    max_tokens=8192 if truncated else _EXTRACT_MAX_TOKENS))
                continue
            prov = _provenance(doc, sec)
            clean: list[dict] = []
            for row in rows:
                q = str(row.get("query", "")).strip()
                a = str(row.get("answer", "")).strip()
                if not q and not a:
                    continue
                clean.append({"query": q, "answer": a, "truncated": False})
                output.append(QAUnit(query=q, answer=a, sources=[prov]))
            if cache is not None and ck:
                cache.put("extract", ck, clean)
    return output


def _qa_from_dict(d: dict, prov: Provenance) -> QAUnit:
    return QAUnit(query=d.get("query", ""), answer=d.get("answer", ""),
                  sources=[prov], truncated=bool(d.get("truncated", False)))


def extract_sop(doc: Document, sec: Section, llm: LLMClient,
                model: str | None = None, cache=None,
                max_tokens: int = _SOP_MAX_TOKENS) -> SOPUnit | None:
    """Clean a section into a whole-Markdown SOP + entry questions. Cached by
    (section content + prompt version) so a re-run skips done sections."""
    images = _section_images(sec)
    user = build_sop_user(sec.full_title, sec.title, mask(sec.body),
                          build_images_block(images))
    prov = _provenance(doc, sec)
    ck = None
    if cache is not None:
        ck = key_for(_content_sha(user), model or "task:sop", SOP_VERSION)
        hit = cache.get("sop", ck)
        if hit is not None:
            return SOPUnit(title=sec.title, markdown=hit.get("markdown", ""),
                           entry_questions=hit.get("entry_questions", []),
                           sources=[prov], truncated=False) if hit.get("markdown") else None
    # Try at the base budget; if the result is truncated or unparseable (a dense
    # section overflowing the completion window), retry once with a bigger budget
    # before giving up. This is the SOP analogue of the QA truncation retry.
    budgets = [max_tokens]
    if _SOP_RETRY_MAX_TOKENS > max_tokens:
        budgets.append(_SOP_RETRY_MAX_TOKENS)
    obj = None
    truncated = False
    last_reason = "no response"
    for attempt, budget in enumerate(budgets):
        try:
            r = llm.complete(system=SOP_SYSTEM, user=user,
                             max_tokens=budget, task="sop", model=model)
        except LLMError as exc:
            raise LLMError(f"extract_sop failed [{doc.topic} {sec.sid}]: {exc}") from exc
        truncated = r.finish_reason == "length"
        obj = parse_json_object(r.text)
        if obj is not None and not truncated:
            break  # clean parse, complete response
        last_reason = "truncated" if truncated else "invalid JSON"
        if attempt + 1 < len(budgets):
            log.warning("extract_sop %s [%s %s] at %d tokens; retrying at %d",
                        last_reason, doc.topic, sec.sid, budget, budgets[attempt + 1])
    if obj is None:
        raise LLMError(
            f"extract_sop returned invalid JSON after retry [{doc.topic} {sec.sid}]")
    md = str(obj.get("markdown", "")).strip()
    eqs = [str(q).strip() for q in obj.get("entry_questions", []) if str(q).strip()]
    if not md:
        return None
    if cache is not None and ck and not truncated:
        cache.put("sop", ck, {"markdown": md, "entry_questions": eqs})
    return SOPUnit(title=sec.title, markdown=md, entry_questions=eqs,
                   sources=[prov], truncated=truncated)


def _content_sha(text: str) -> str:
    import hashlib
    return hashlib.sha256(text.encode("utf-8")).hexdigest()
