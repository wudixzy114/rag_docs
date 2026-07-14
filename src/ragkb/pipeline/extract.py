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
    build_extract_user, build_sop_user, build_images_block,
)
from ragkb.pipeline.scrub import mask
from ragkb.pipeline.units import Provenance, QAUnit, SOPUnit
from ragkb.store.cache import key_for

log = logging.getLogger(__name__)

_EXTRACT_MAX_TOKENS = 4096
_SOP_MAX_TOKENS = 6144


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
    return Provenance(
        topic=doc.topic, doc_title=doc.title,
        heading_path=sec.full_title, section_sid=sec.sid,
        source_sha=doc.source_sha,
        image_refs=[im.rel_path for im in sec.images],
    )


def extract_qa(doc: Document, sec: Section, llm: LLMClient,
               model: str | None = None, cache=None) -> list[QAUnit]:
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
                         max_tokens=_EXTRACT_MAX_TOKENS, task="extract", model=model)
    except LLMError as exc:
        log.warning("extract_qa failed [%s %s]: %s", doc.topic, sec.sid, exc)
        return []
    truncated = r.finish_reason == "length"
    arr = parse_json_array(r.text) or []
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


def _qa_from_dict(d: dict, prov: Provenance) -> QAUnit:
    return QAUnit(query=d.get("query", ""), answer=d.get("answer", ""),
                  sources=[prov], truncated=bool(d.get("truncated", False)))


def extract_sop(doc: Document, sec: Section, llm: LLMClient,
                model: str | None = None, cache=None) -> SOPUnit | None:
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
    try:
        r = llm.complete(system=SOP_SYSTEM, user=user,
                         max_tokens=_SOP_MAX_TOKENS, task="sop", model=model)
    except LLMError as exc:
        log.warning("extract_sop failed [%s %s]: %s", doc.topic, sec.sid, exc)
        return None
    truncated = r.finish_reason == "length"
    obj = parse_json_object(r.text) or {}
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
