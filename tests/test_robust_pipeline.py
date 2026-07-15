from __future__ import annotations

import zipfile
import threading
import time
from concurrent.futures import ThreadPoolExecutor
import pytest
from pathlib import Path

from ragkb.config import LLMSettings
from ragkb.llm.client import LLMResult, LLMUsage, LLMClient
from ragkb.parse.markdown import parse_document
from ragkb.parse.source import load_source, source_bundle_sha
from ragkb.pipeline.batching import pack_by_size
from ragkb.pipeline.gate_semantic import review_qa
from ragkb.pipeline.extract import extract_qa
from ragkb.pipeline.regenerate import review_with_regeneration
from ragkb.pipeline.orchestrator import discover_topics
from ragkb.pipeline.sections import split_oversize_sections
from ragkb.pipeline.units import Provenance, QAUnit
from ragkb.server.app import _public_results


def test_headerless_document_is_not_dropped(tmp_path):
    path = tmp_path / "loose-note.txt"
    path.write_text("第一段没有标题。\n\n第二段仍然是有效知识。", "utf-8")
    doc = parse_document(path, topic="loose")
    sections = list(doc.iter_sections())
    assert len(sections) == 1
    assert sections[0].title == "loose-note"
    assert "第一段" in sections[0].body and "第二段" in sections[0].body


def test_preamble_is_kept_without_replacing_document_title(tmp_path):
    path = tmp_path / "guide.md"
    path.write_text("前言内容\n\n# 正式标题\n正文", "utf-8")
    doc = parse_document(path, topic="guide")
    assert doc.title == "正式标题"
    sections = list(doc.iter_sections())
    assert sections[0].sid == "0" and "前言内容" in sections[0].body


def test_recursive_discovery_accepts_arbitrary_names_and_formats(tmp_path):
    nested = tmp_path / "a" / "b"
    nested.mkdir(parents=True)
    for name in ("guide.md", "notes.txt", "page.html", "manual.docx", "book.pdf"):
        (nested / name).write_bytes(b"x")
    (nested / "ignored.csv").write_text("x", "utf-8")
    hidden = tmp_path / ".cache"
    hidden.mkdir()
    (hidden / "secret.md").write_text("x", "utf-8")
    found = {p.name for p in discover_topics(tmp_path)}
    assert found == {"guide.md", "notes.txt", "page.html", "manual.docx", "book.pdf"}


def test_source_hash_changes_when_referenced_assets_change(tmp_path):
    source = tmp_path / "guide.md"
    source.write_text("# Guide\n![](images/screen.png)", "utf-8")
    images = tmp_path / "images"
    images.mkdir()
    image = images / "screen.png"
    image.write_bytes(b"first")
    before = source_bundle_sha(source)
    image.write_bytes(b"second")
    assert source_bundle_sha(source) != before


def test_html_and_docx_conversion_preserve_structure(tmp_path):
    html = tmp_path / "page.html"
    html.write_text("<h1>标题</h1><p>正文</p><h2>步骤</h2><ul><li>执行命令</li></ul>", "utf-8")
    parsed = parse_document(html, topic="html")
    assert [s.title for s in parsed.iter_sections()] == ["标题", "步骤"]

    docx = tmp_path / "manual.docx"
    xml = '''<?xml version="1.0" encoding="UTF-8"?>
    <w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">
      <w:body><w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>手册</w:t></w:r></w:p>
      <w:p><w:r><w:t>操作内容</w:t></w:r></w:p></w:body></w:document>'''.encode("utf-8")
    with zipfile.ZipFile(docx, "w") as archive:
        archive.writestr("word/document.xml", xml)
    text = load_source(docx)
    assert "# 手册" in text and "操作内容" in text


def test_size_batching_and_long_section_split_keep_fences():
    assert [len(x) for x in pack_by_size(["aa", "bb", "cccc"], len,
                                          max_items=2, max_chars=5)] == [2, 1]
    from ragkb.parse.model import Section
    body = "段落一" * 30 + "\n\n```bash\n" + "echo x\n" * 20 + "```\n\n" + "段落二" * 30
    parts = split_oversize_sections([Section(level=1, title="长文", body=body, sid="1")],
                                    max_chars=100)
    assert len(parts) >= 2
    assert all(p.body.count("```") % 2 == 0 for p in parts)


class _ReviewLLM:
    def __init__(self):
        self.calls: list[str] = []

    def complete(self, **kwargs):
        self.calls.append(kwargs["user"])
        if len(self.calls) == 1:
            return LLMResult(text='[{"id":0,"verdict":"pass","reason":"ok",'
                                   '"valid_paraphrases":["安全问法"]}]')
        return LLMResult(text='[{"id":1,"verdict":"revise","reason":"补全",'
                               '"revised_answer":"按原文执行完整步骤"}]')


def test_review_repairs_missing_ids_and_routes_revision_to_regeneration():
    source = Provenance(heading_path="章节", source_excerpt="原始证据：按原文执行完整步骤")
    units = [QAUnit(query="问题一", answer="答案一", paraphrases=["安全问法", "新增场景"],
                    sources=[source]),
             QAUnit(query="问题二", answer="不完整", sources=[source])]
    llm = _ReviewLLM()
    review_qa(units, llm)
    assert len(llm.calls) == 2
    assert "原始证据" in llm.calls[0]
    assert units[0].semantic_ok is True
    assert units[0].paraphrases == ["安全问法"]
    assert units[1].semantic_ok is False
    assert units[1].publication_status == "failed_review"
    assert units[1].answer == "不完整"


class _MissingRevisionLLM:
    def complete(self, **kwargs):
        return LLMResult(text='[{"id":0,"verdict":"revise","reason":"需要修改",'
                               '"valid_paraphrases":[]}]')


def test_review_rejects_revise_without_actual_revision():
    unit = QAUnit(query="问题", answer="答案",
                  sources=[Provenance(source_excerpt="原文")])
    review_qa([unit], _MissingRevisionLLM())
    assert unit.semantic_ok is False
    assert unit.semantic_reason == "revise:需要修改"


class _ReviewRegenerateLLM:
    def __init__(self, final_verdict="pass"):
        self.review_calls = 0
        self.final_verdict = final_verdict

    def complete(self, **kwargs):
        if "问答修复专家" in kwargs["system"]:
            return LLMResult(text='[{"id":0,"query":"如何完整处理？",'
                                   '"answer":"第一步执行 A，第二步执行 B。"}]')
        self.review_calls += 1
        verdict = "reject" if self.review_calls == 1 else self.final_verdict
        return LLMResult(text=f'[{{"id":0,"verdict":"{verdict}",'
                               '"reason":"check","valid_paraphrases":[]}]')


def test_review_regenerates_once_then_approves():
    unit = QAUnit(query="怎么做？", answer="只执行 A",
                  sources=[Provenance(topic="t", source_excerpt="第一步执行 A，第二步执行 B。")])
    review_with_regeneration([unit], _ReviewRegenerateLLM(), max_attempts=1)
    assert unit.semantic_ok is True
    assert unit.publication_status == "approved"
    assert unit.review_attempts == 1
    assert unit.review_history == ["reject:check"]
    assert "第二步" in unit.answer


def test_review_failure_after_one_regeneration_is_retained_and_marked():
    unit = QAUnit(query="怎么做？", answer="只执行 A",
                  sources=[Provenance(topic="t", source_excerpt="第一步执行 A，第二步执行 B。")])
    review_with_regeneration(
        [unit], _ReviewRegenerateLLM(final_verdict="reject"), max_attempts=1)
    assert unit.semantic_ok is False
    assert unit.publication_status == "failed_review"
    assert unit.needs_review is True
    assert unit.review_attempts == 1


class _InvalidJSONLLM:
    def complete(self, **kwargs):
        return LLMResult(text="not json")


def test_extraction_failure_is_not_silently_treated_as_empty(tmp_path):
    path = tmp_path / "source.md"
    path.write_text("# 问题\n正文", "utf-8")
    doc = parse_document(path, topic="t")
    section = next(doc.iter_sections())
    with pytest.raises(Exception, match="invalid JSON"):
        extract_qa(doc, section, _InvalidJSONLLM())


def test_gemini_defaults_route_fast_and_complex_tasks():
    settings = LLMSettings(
        XIAOSHU_MODEL="Gemini-3.1-Pro-Preview-joybuilder",
        JD_LLM_TASK_MODELS="", JD_LLM_FALLBACK_MODELS="",
        JD_LLM_SIMPLE_MODELS="")
    assert settings.model_for("classify").startswith("Gemini-3-Flash")
    assert settings.model_for("paraphrase").startswith("Gemini-3-Flash")
    assert settings.model_for("extract").startswith("Gemini-3.1-Pro")
    assert settings.model_for("review").startswith("Gemini-3.1-Pro")


class _ConcurrencyLLM(LLMClient):
    def __init__(self, settings):
        super().__init__(settings=settings, client=object())
        self.current = 0
        self.peak = 0
        self.counter_lock = threading.Lock()

    def _post_chat(self, payload):
        with self.counter_lock:
            self.current += 1
            self.peak = max(self.peak, self.current)
        time.sleep(0.01)
        with self.counter_lock:
            self.current -= 1
        return LLMResult(text="ok", usage=LLMUsage(calls=1))


def test_llm_client_enforces_global_concurrency_limit():
    settings = LLMSettings(
        XIAOSHU_MODEL="Test-OpenAI", JD_LLM_FALLBACK_MODELS="",
        JD_LLM_TASK_MODELS="", JD_LLM_MAX_CONCURRENCY=2)
    client = _ConcurrencyLLM(settings)
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda _: client.complete(system="s", user="u", model="Test-OpenAI"),
                      range(8)))
    assert client.peak == 2
    assert client.total_usage.calls == 8


def test_public_state_strips_source_evidence_but_keeps_review_status():
    data = {"qa": [{"query": "q", "publication_status": "failed_review",
                    "sources": [{"topic": "t", "source_excerpt": "large evidence"}]}],
            "sop": []}
    public = _public_results(data)
    assert public["qa"][0]["publication_status"] == "failed_review"
    assert public["qa"][0]["sources"] == [{"topic": "t"}]
