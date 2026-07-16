"""复审 results.json 里 failed_review 的 SOP：能修的修、修不好的丢，然后重导出。

背景：全量重跑后有 11 条 SOP 处于 failed_review——部分是 no_verdict（审核当时
没拿到判定，内容未必差），部分是 revise（审核指出编造/遗漏/字段错，可源文重生成修正）。
用户要求：全部再审一遍，有必要就补，没必要就丢。

做法（复用真实流水线逻辑，保证一致）：
1. load_results 读回全部 QA/SOP（含 source_excerpt + redaction map）。
2. 隔离 failed_review 的 SOP，重置 review 状态（review_attempts=0, semantic_ok=None），
   使其获得一次完整的 复审 + 一次 source-grounded 重生成 + 再复审。
3. 复审后 approved 的写回、仍 failed 的丢弃。
4. 覆写 results.json（approved SOP 全集 + 原 QA 全集不动），再 export 打最终包。

运行： uv run python scripts/rehabilitate_failed_sop.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ragkb.config import get_settings
from ragkb.llm.client import LLMClient
from ragkb.pipeline.export import load_results, export_all
from ragkb.pipeline.regenerate import review_sop_with_regeneration


def main() -> None:
    settings = get_settings()
    out = settings.output_dir
    qa, sop = load_results(out)

    failed = [u for u in sop if u.publication_status == "failed_review"]
    approved = [u for u in sop if u.publication_status != "failed_review"]
    print(f"载入：QA {len(qa)}，SOP {len(sop)}（approved {len(approved)}，"
          f"failed_review {len(failed)}）")
    if not failed:
        print("没有 failed_review SOP，无需复审。")
        return

    # 重置 review 状态，让它们获得完整的 复审 + 重生成 循环。
    for u in failed:
        u.semantic_ok = None
        u.review_attempts = 0
        u.needs_review = False
        u.publication_status = "pending"

    llm = LLMClient()
    print(f"\n开始复审 {len(failed)} 条（并发审核 + 至多一次 source-grounded 重生成）…")
    review_sop_with_regeneration(failed, llm, max_attempts=1)

    rescued = [u for u in failed if u.semantic_ok and u.publication_status == "approved"]
    dropped = [u for u in failed if u not in rescued]
    print(f"\n复审结果：救回 {len(rescued)} 条，丢弃 {len(dropped)} 条。")
    print("\n--- 救回 ---")
    for u in rescued:
        t = u.sources[0].topic if u.sources else "?"
        print(f"  ✓ {t} / {u.title}")
    print("\n--- 丢弃（复审仍不达标）---")
    for u in dropped:
        t = u.sources[0].topic if u.sources else "?"
        print(f"  ✗ {t} / {u.title}  ← {u.semantic_reason}")

    # 写回 results.json：QA 原样 + (原 approved SOP + 新救回的)。丢弃的不写。
    final_sop = approved + rescued
    _persist(out, qa, final_sop)
    print(f"\nresults.json 已更新：QA {len(qa)}，SOP {len(final_sop)}"
          f"（较前 {'+' if len(rescued) else ''}{len(rescued)}）")

    # 重新导出最终产物。
    stats = export_all(qa, final_sop, out, include_paraphrases=False)
    print(f"\n已导出最终产物：QA单元 {stats.qa_units}，检索行 {stats.qa_rows}，"
          f"SOP {stats.sop_files}，待审 {stats.needs_review}，模块 {stats.modules}")


def _persist(out: Path, qa, sop) -> None:
    """按 orchestrator._persist_results 的同款结构覆写 results.json。"""
    payload = {"qa": [], "sop": []}
    seen = set()
    for u in qa:
        mod = u.sources[0].topic if u.sources else ""
        key = (mod, u.publication_status, u.unit_id)
        if key in seen:
            continue
        seen.add(key)
        payload["qa"].append({
            "query": u.query, "answer": u.answer, "paraphrases": u.paraphrases,
            "needs_review": u.needs_review, "semantic_reason": u.semantic_reason,
            "semantic_ok": u.semantic_ok, "review_attempts": u.review_attempts,
            "publication_status": u.publication_status,
            "review_history": u.review_history,
            "sources": [s.to_dict(include_excerpt=True) for s in u.sources]})
    for u in sop:
        payload["sop"].append({
            "title": u.title, "markdown": u.markdown,
            "entry_questions": u.entry_questions, "struct_ok": u.struct_ok,
            "struct_reason": u.struct_reason, "semantic_ok": u.semantic_ok,
            "semantic_reason": u.semantic_reason, "needs_review": u.needs_review,
            "review_attempts": u.review_attempts,
            "publication_status": u.publication_status,
            "review_history": u.review_history,
            "sources": [s.to_dict(include_excerpt=True) for s in u.sources]})
    p = out / "results.json"
    tmp = p.with_suffix(".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(p)


if __name__ == "__main__":
    main()
