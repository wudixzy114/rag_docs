"""Deterministic-core tests — no LLM, no network. These lock in the correctness
guarantees the user cares most about: no scrambled sections, no truncated/empty
units slipping through, clean 2-column CSV."""
from __future__ import annotations

from pathlib import Path

from ragkb.parse.markdown import parse_document
from ragkb.pipeline.dedup import dedup_qa
from ragkb.pipeline.export import export_all
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
    a = QAUnit(query="任务OOM怎么办", answer="加内存", sources=[Provenance(topic="t")])
    b = QAUnit(query="任务 OOM 怎么办？", answer="加内存",   # near-dup query, same module
               sources=[Provenance(topic="t")])
    b.paraphrases = ["内存爆了咋整"]
    kept = dedup_qa([a, b])
    assert len(kept) == 1
    assert "内存爆了咋整" in kept[0].query_keys()          # dropped unit's key preserved


def test_dedup_is_module_scoped():
    """Same question in two modules must NOT merge — each module keeps its own."""
    a = QAUnit(query="clone没权限怎么办", answer="配置SSH密钥(LLM版)",
               sources=[Provenance(topic="09_9N-LLM")])
    b = QAUnit(query="clone没权限怎么办", answer="配置SSH密钥(Tritium版)",
               sources=[Provenance(topic="10_9N-Tritium")])
    kept = dedup_qa([a, b])
    assert len(kept) == 2                                 # cross-module dupes kept


def test_export_csv_three_columns_with_module(tmp_path):
    u = QAUnit(query="含,逗号和\n换行的问题", answer='答案含"引号"和,逗号',
               sources=[Provenance(topic="05_9Nctl常见问题汇总")])
    u.paraphrases = ["另一个问法"]
    stats = export_all([u], [], tmp_path)
    import csv
    rows = list(csv.reader(open(tmp_path / "qa_pairs.csv", encoding="utf-8-sig")))
    assert rows[0] == ["Query", "Answer", "Module"]
    assert all(len(r) == 3 for r in rows)                 # never misaligned
    assert stats.qa_rows == 2                             # main query + 1 paraphrase
    assert all(r[2] == "05_9Nctl常见问题汇总" for r in rows[1:])   # module column filled
    # per-module partition written too
    assert (tmp_path / "by_module" / "05_9Nctl常见问题汇总" / "qa_pairs.csv").is_file()


def test_reversible_mask_restore():
    from ragkb.pipeline.scrub import Redactor
    r = Redactor()
    orig = "网卡 MAC a2:16:80:bb:aa:2b, 内网 10.20.30.40, 手机 13021946001, key 9de5dee757784f6abc44a43c93c11d20"
    masked = r.mask(orig)
    # masked form must NOT contain the raw sensitive values (so it passes the gateway)
    assert "a2:16:80:bb:aa:2b" not in masked
    assert "10.20.30.40" not in masked
    assert "13021946001" not in masked
    # deterministic: same value → same token across calls
    assert r.mask("a2:16:80:bb:aa:2b") == r.mask("a2:16:80:bb:aa:2b")
    # restore round-trips back to the exact original (internal KB shows real values)
    assert r.restore(masked) == orig


def test_mask_is_stable_for_dedup():
    """Same value must map to the same token so masked text stays dedup-stable."""
    from ragkb.pipeline.scrub import Redactor
    r = Redactor()
    a = r.mask("连接 10.1.2.3 失败")
    b = r.mask("连接 10.1.2.3 超时")
    # the IP placeholder is identical in both
    tok = a.replace("连接 ", "").replace(" 失败", "")
    assert tok in b


def test_mask_covers_mac_ip_phone():
    """The gateway content-filter 400s on these; pre-send mask must remove them."""
    from ragkb.pipeline.scrub import Redactor
    r = Redactor()
    s = r.mask("网卡 a2:16:80:bb:aa:2b, 内网 10.20.30.40 与 192.168.1.1, 手机 13021946001, 链路 fe80::ecee:eeff:feee")
    assert "a2:16:80:bb:aa:2b" not in s
    assert "10.20.30.40" not in s and "192.168.1.1" not in s
    assert "13021946001" not in s
    assert "fe80::ecee" not in s
    # A public IP is NOT masked (sometimes legitimate example content).
    assert "8.8.8.8" in r.mask("dns 8.8.8.8")


def test_content_blocked_detector():
    from ragkb.llm.client import _is_content_blocked
    body = '{"error":{"code":400,"message":"sensitive contain:[\\"MAC地址\\"]","status":"FAILED_PRECONDITION"}}'
    assert _is_content_blocked(400, body)
    assert not _is_content_blocked(200, body)
    assert not _is_content_blocked(400, '{"error":"some other 400"}')
