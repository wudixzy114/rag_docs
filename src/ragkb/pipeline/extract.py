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
    EXTRACT_SYSTEM, SOP_SYSTEM, build_extract_user, build_sop_user,
    build_images_block,
)
from ragkb.pipeline.scrub import scrub
from ragkb.pipeline.units import Provenance, QAUnit, SOPUnit

log = logging.getLogger(__name__)

_EXTRACT_MAX_TOKENS = 4096
_SOP_MAX_TOKENS = 6144


def _section_images(sec: Section) -> list[dict]:
    """Vision transcripts for the images in this section (own images only).
    Scrubbed pre-send so gateway content-safety doesn't 400 on MAC/IP in a chart."""
    out = []
    for im in sec.images:
        out.append({"rel_path": im.rel_path,
                    "transcript": scrub(im.vision_text or im.inline_ocr),
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
               model: str | None = None) -> list[QAUnit]:
    """Extract QA pairs from a section. Truncation (finish_reason==length) is
    recorded on every unit so the struct gate can fail them and the orchestrator
    can retry with a larger budget."""
    images = _section_images(sec)
    user = build_extract_user(sec.full_title, sec.title, scrub(sec.body),
                              build_images_block(images))
    try:
        r = llm.complete(system=EXTRACT_SYSTEM, user=user,
                         max_tokens=_EXTRACT_MAX_TOKENS, task="extract", model=model)
    except LLMError as exc:
        log.warning("extract_qa failed [%s %s]: %s", doc.topic, sec.sid, exc)
        return []
    truncated = r.finish_reason == "length"
    arr = parse_json_array(r.text) or []
    prov = _provenance(doc, sec)
    units: list[QAUnit] = []
    for el in arr:
        if not isinstance(el, dict):
            continue
        q, a = str(el.get("query", "")).strip(), str(el.get("answer", "")).strip()
        if not q and not a:
            continue
        units.append(QAUnit(query=q, answer=a, sources=[prov], truncated=truncated))
    return units


def extract_sop(doc: Document, sec: Section, llm: LLMClient,
                model: str | None = None) -> SOPUnit | None:
    """Clean a section into a whole-Markdown SOP + entry questions."""
    images = _section_images(sec)
    user = build_sop_user(sec.full_title, sec.title, scrub(sec.body),
                          build_images_block(images))
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
    return SOPUnit(title=sec.title, markdown=md, entry_questions=eqs,
                   sources=[_provenance(doc, sec)], truncated=truncated)
