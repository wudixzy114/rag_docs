"""Deterministic-core tests — no LLM, no network. These lock in the correctness
guarantees the user cares most about: no scrambled sections, no truncated/empty
units slipping through, clean 2-column CSV."""
from __future__ import annotations

from pathlib import Path

from ragkb.parse.markdown import parse_document
from ragkb.pipeline.dedup import dedup_qa
from ragkb.pipeline.export import export_all, scrub
from ragkb.pipeline.gate_struct import gate_qa, gate_sop
from ragkb.pipeline.units import Provenance, QAUnit, SOPUnit


def _write(tmp: Path, text: str) -> Path:
    d = tmp / "01_topic"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "原始文档.md"
    p.write_text(text, "utf-8")
    return p


def test_fence_aware_headings(tmp_path):
    """A '# comment' inside a code fence must NOT become a heading/section."""
    md = "# 真标题\n\n正文\n\n```bash\n# 这是代码注释不是标题\npip install x\n```\n\n## 子标题\n更多正文\n"
    doc = parse_document(_write(tmp_path, md))
    titles = [s.title for s in doc.iter_sections()]
    assert "真标题" in titles
    assert "子标题" in titles
    assert "这是代码注释不是标题" not in titles       # the critical anti-错乱 guarantee
    # The code comment stays in the parent section body.
    root_sec = next(s for s in doc.iter_sections() if s.title == "真标题")
    assert "pip install x" in root_sec.body


def test_ocr_block_lifted_from_body(tmp_path):
    md = ('# 标题\n\n![](images/img-01.png)\n\n<!-- ocr-source: images/img-01.png -->\n'
          '<details>\n<summary>图片文字识别（OCR）</summary>\n\n<pre>OCR文字内容</pre>\n</details>\n\n正文段落\n')
    doc = parse_document(_write(tmp_path, md))
    sec = next(s for s in doc.iter_sections() if s.title == "标题")
    assert len(sec.images) == 1
    assert sec.images[0].inline_ocr == "OCR文字内容"
    assert "OCR文字内容" not in sec.body            # raw OCR scaffolding stripped from prose
    assert "正文段落" in sec.body


def test_struct_gate_rejects_truncation_and_empty():
    good = gate_qa(QAUnit(query="怎么办", answer="这样做即可"))
    assert good.struct_ok
    trunc = gate_qa(QAUnit(query="q", answer="半截答案", truncated=True))
    assert not trunc.struct_ok and "truncated" in trunc.struct_reason
    empty = gate_qa(QAUnit(query="q", answer="  "))
    assert not empty.struct_ok
    fence = gate_qa(QAUnit(query="q", answer="```python\nx=1"))   # unbalanced fence
    assert not fence.struct_ok


def test_struct_gate_does_not_reject_short_answers():
    """Layer1 must not impose a length floor — a short correct answer passes."""
    assert gate_qa(QAUnit(query="端口是多少", answer="8080")).struct_ok


def test_sop_gate_requires_entry_questions():
    assert not gate_sop(SOPUnit(title="t", markdown="# t\n步骤", entry_questions=[])).struct_ok
    assert gate_sop(SOPUnit(title="t", markdown="# t\n步骤", entry_questions=["怎么做"])).struct_ok


def test_dedup_merges_and_unions_keys():
    a = QAUnit(query="任务OOM怎么办", answer="加内存")
    b = QAUnit(query="任务 OOM 怎么办？", answer="加内存")   # near-dup query
    b.paraphrases = ["内存爆了咋整"]
    kept = dedup_qa([a, b])
    assert len(kept) == 1
    assert "内存爆了咋整" in kept[0].query_keys()          # dropped unit's key preserved


def test_export_csv_two_columns_quote_all(tmp_path):
    u = QAUnit(query="含,逗号和\n换行的问题", answer='答案含"引号"和,逗号',
               sources=[Provenance(topic="t")])
    u.paraphrases = ["另一个问法"]
    stats = export_all([u], [], tmp_path)
    import csv
    rows = list(csv.reader(open(tmp_path / "qa_pairs.csv", encoding="utf-8-sig")))
    assert rows[0] == ["Query", "Answer"]
    assert all(len(r) == 2 for r in rows)                 # never misaligned
    assert stats.qa_rows == 2                             # main query + 1 paraphrase
    assert not any((not r[0].strip() or not r[1].strip()) for r in rows[1:])


def test_scrub_redacts_secrets_but_keeps_hostnames():
    assert "[REDACTED]" in scrub("Authorization: Bearer abc123def456")
    assert "[REDACTED]" in scrub("9de5dee757784f6abc44a43c93c11d20")   # 32-hex key
    assert "storage.jd.local" in scrub("wget http://storage.jd.local/x.sh")  # host kept
