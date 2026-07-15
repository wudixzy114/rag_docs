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


def test_image_directive_glued_to_details_close(tmp_path):
    """An image directive glued onto the previous OCR block's `</details>` line
    (`</details>![](img-10.png)`) must still be parsed — the OCR-block consumer
    must not swallow the trailing image. Regression: img-10 was silently dropped."""
    md = ("# 标题\n\n![](images/img-09.png)\n\n"
          "<!-- ocr-source: images/img-09.png -->\n<details>\n<pre>GPU图</pre>"
          "</details>![](images/img-10.png)\n\n"
          "<!-- ocr-source: images/img-10.png -->\n<details>\n<pre>CPU图</pre>\n</details>\n\n正文\n")
    doc = parse_document(_write(tmp_path, md))
    paths = [im.rel_path for im in doc.all_images()]
    assert "images/img-09.png" in paths and "images/img-10.png" in paths   # both survive
    sec = next(s for s in doc.iter_sections() if s.title == "标题")
    ocr = {im.rel_path: im.inline_ocr for im in sec.images}
    assert ocr["images/img-09.png"] == "GPU图" and ocr["images/img-10.png"] == "CPU图"  # not misrouted


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


def test_sop_filename_is_module_plus_section_number(tmp_path):
    """SOP filenames drop the long title and key on the section NUMBER (upload
    target caps path length). Long descriptive titles must not blow up the name."""
    long_title = ("7.4.7. 关闭 SampleService 断点续训时，同时关闭 sample dispatcher "
                  "的 checkpoint 配置以避免状态不一致")
    u = SOPUnit(title=long_title, markdown="# x\n步骤", entry_questions=["怎么做"],
                sources=[Provenance(topic="10_9N-Tritium")])
    export_all([], [u], tmp_path)
    files = list((tmp_path / "sop").glob("*.md"))
    assert len(files) == 1
    assert files[0].name == "10_9N-Tritium__7.4.7.md"
    assert len(files[0].stem) <= 25          # was 80+ under the old title-based scheme


def test_entry_questions_injected_inside_first_heading(tmp_path):
    """Entry-questions must live INSIDE the doc (after the first `#`), never as a
    pre-heading block — else the vector service chunks them into an answer-less
    orphan chunk. Guarantee: nothing precedes the `#`, and the 问法 line sits in
    the same (first) chunk as the title."""
    md = "# 代理服务的配置\n\n在集群中要访问 tensorboard 需先配置代理。\n\n## 步骤\n1. 打开插件"
    u = SOPUnit(title="4.1.1. 代理服务的配置", markdown=md,
                entry_questions=["tensorboard 访问不了？", "怎么配代理？"],
                sources=[Provenance(topic="01_模型训练常见问题自查")])
    export_all([], [u], tmp_path)
    text = (tmp_path / "sop" / "01_模型训练常见问题自查__4.1.1.md").read_text("utf-8")
    # The document opens with the H1 — no orphan block before it.
    assert text.lstrip().startswith("# 代理服务的配置")
    assert "<!-- entry-questions" not in text          # old orphan format gone
    # 问法 appears after the H1 but before the first ## (i.e. in the first chunk).
    i_h1 = text.index("# 代理服务的配置")
    i_q = text.index("常见问法")
    i_h2 = text.index("## 步骤")
    assert i_h1 < i_q < i_h2
    assert "tensorboard 访问不了？" in text and "怎么配代理？" in text


def test_sop_filename_fallback_and_collision(tmp_path):
    """Unnumbered pages get a compact FAQ/slug; a repeated key never overwrites."""
    faq = SOPUnit(title="【FAQ】任务优先级规则说明", markdown="# a\n内容",
                  entry_questions=["怎么排优先级"], sources=[Provenance(topic="03_任务优先级规则说明")])
    dup1 = SOPUnit(title="1. 步骤", markdown="# b\n内容一", entry_questions=["q1"],
                   sources=[Provenance(topic="06_批调度")])
    dup2 = SOPUnit(title="1. 另一节", markdown="# c\n内容二", entry_questions=["q2"],
                   sources=[Provenance(topic="06_批调度")])   # same module+number → collision
    export_all([], [faq, dup1, dup2], tmp_path)
    names = {p.name for p in (tmp_path / "sop").glob("*.md")}
    assert "03_任务优先级规则说明__FAQ.md" in names
    # both same-key SOPs survive (one suffixed), neither silently lost
    assert "06_批调度__1.md" in names and "06_批调度__1-2.md" in names


def test_upload_zip_utf8_and_no_dsstore(tmp_path):
    """The zip must set the UTF-8 filename flag on CJK names (else mojibake on
    unzip) and never bundle .DS_Store."""
    import zipfile
    u = QAUnit(query="问题", answer="答案", sources=[Provenance(topic="05_9Nctl常见问题汇总")])
    s = SOPUnit(title="1. 标题", markdown="# x\n步骤", entry_questions=["怎么做"],
                sources=[Provenance(topic="05_9Nctl常见问题汇总")])
    (tmp_path / ".DS_Store").write_bytes(b"junk")     # must be excluded
    export_all([u], [s], tmp_path)
    zpath = tmp_path / "知识库上传包.zip"
    assert zpath.is_file()
    z = zipfile.ZipFile(zpath)
    infos = z.infolist()
    assert not any(".DS_Store" in i.filename for i in infos)
    cjk = [i for i in infos if not i.filename.isascii()]
    assert cjk and all(i.flag_bits & 0x800 for i in cjk)   # every CJK name flagged UTF-8


def test_export_round_trips_through_results_json(tmp_path):
    """load_results(results.json) → export_all must reproduce the same unit set,
    including cross-module twins (the persist dedup keys on (module, unit_id))."""
    import json
    from ragkb.pipeline.export import load_results
    payload = {"qa": [
        {"query": "clone没权限怎么办", "answer": "配置SSH(LLM)", "paraphrases": ["拉代码没权限"],
         "needs_review": False, "semantic_reason": "", "sources": [{"topic": "09_9N-LLM"}]},
        {"query": "clone没权限怎么办", "answer": "配置SSH(Tritium)", "paraphrases": [],
         "needs_review": False, "semantic_reason": "", "sources": [{"topic": "10_9N-Tritium"}]},
    ], "sop": []}
    (tmp_path / "results.json").write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    qa, sop = load_results(tmp_path)
    assert len(qa) == 2                    # cross-module twins both survive rehydration
    assert {u.sources[0].topic for u in qa} == {"09_9N-LLM", "10_9N-Tritium"}


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
