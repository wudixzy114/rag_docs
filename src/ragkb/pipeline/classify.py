"""Section-level classification: qa | sop | skip.

Two-tier, to save tokens (the plan's cost lever): a deterministic pre-filter
labels the obvious cases by heading/body cues; only genuinely ambiguous sections
go to the LLM (batched, indexed — the verify.py pattern). Grounded in the real
material's cues (如何定位/如何解决/报错 → qa; 步骤/流程/规则/板斧 → sop; 目录/提问之前 → skip).
"""
from __future__ import annotations

import logging
import re

from ragkb.llm.client import LLMClient, LLMError
from ragkb.parse.model import Section
from ragkb.pipeline.jsonutil import parse_json_array
from ragkb.pipeline.prompts import CLASSIFY_SYSTEM, build_classify_user

log = logging.getLogger(__name__)

_CLASSIFY_BATCH = 25

# Heading cues (checked on the title). QA: a symptom/problem/question. SOP: an
# ordered procedure / rule / concept.
_QA_CUES = re.compile(r"(如何|怎么|怎样|为什么|为何|是什么|报错|失败|错误|异常|问题|"
                      r"排查|定位|超时|不了|无法|起不来|hang|oom|error|\?|？)", re.I)
_SOP_CUES = re.compile(r"(流程|步骤|规则|说明|板斧|规范|介绍|概述|原理|机制|准备|"
                       r"配置方法|操作步骤)")
_SKIP_CUES = re.compile(r"^(目录|附录|提问之前|概览|index|toc|-+|=+)\s*$", re.I)


def _prefilter(title: str, body: str) -> str | None:
    """Deterministic label, or None if ambiguous (→ LLM). Only labels when the
    signal is strong and unambiguous; otherwise defer to the model."""
    t = title.strip()
    # Strip leading section numbers ("3.1. ", "1. ") before cue matching.
    t_clean = re.sub(r"^[\d.、]+\s*", "", t).strip()
    if _SKIP_CUES.match(t_clean) or (len(t_clean) <= 1 and len(body) < 20):
        return "skip"
    body_len = len(body.strip())
    if body_len < 15 and not body.strip():
        return "skip"
    qa = bool(_QA_CUES.search(t_clean))
    sop = bool(_SOP_CUES.search(t_clean))
    if qa and not sop:
        return "qa"
    if sop and not qa:
        return "sop"
    return None       # ambiguous or no cue → let the LLM decide


def _section_content(section: Section) -> str:
    """Classification evidence, including authoritative image transcripts.

    Screenshot-only sections are common in operational docs. Treating their
    empty prose body as empty content would silently label them skip.
    """
    image_text = "\n\n".join(
        f"[图片 {im.rel_path}]\n{im.best_text}"
        for im in section.images if im.best_text)
    return f"{section.body}\n\n{image_text}".strip()


def classify_sections(sections: list[Section], llm: LLMClient) -> dict[str, str]:
    """Return {section.sid: label}. Pre-filter first; batch the rest to the LLM.
    On LLM failure, ambiguous sections default to 'qa' (safer than dropping — a
    mis-routed unit is still reviewable; a skipped one is silently lost)."""
    labels: dict[str, str] = {}
    ambiguous: list[Section] = []
    for s in sections:
        pre = _prefilter(s.title, _section_content(s))
        if pre is not None:
            labels[s.sid] = pre
        else:
            ambiguous.append(s)

    if not ambiguous:
        return labels

    id_map = {i: s for i, s in enumerate(ambiguous)}
    for start in range(0, len(ambiguous), _CLASSIFY_BATCH):
        chunk = ambiguous[start:start + _CLASSIFY_BATCH]
        payload = [{"id": start + j, "title": s.title,
                    "body_preview": _section_content(s)} for j, s in enumerate(chunk)]
        try:
            r = llm.complete(system=CLASSIFY_SYSTEM,
                             user=build_classify_user(payload),
                             max_tokens=1500, task="classify")
            arr = parse_json_array(r.text) or []
        except LLMError as exc:
            log.warning("classify batch failed: %s", exc)
            arr = []
        got = {}
        for el in arr:
            if isinstance(el, dict) and "id" in el:
                try:
                    got[int(el["id"])] = str(el.get("label", "")).lower()
                except (TypeError, ValueError):
                    pass
        for j, s in enumerate(chunk):
            label = got.get(start + j, "")
            if label not in ("qa", "sop", "skip"):
                label = "qa"      # fail-safe default: review beats silent loss
            labels[s.sid] = label
    return labels
