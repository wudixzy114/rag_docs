"""给 4 条 no_verdict SOP 一次干净的复审机会。

这 4 条在全量重跑和上一轮复审里都因限流导致审核【没拿到判定】(no_verdict)——
不是内容差，是审核请求在拥塞下丢了 verdict。这里单独、低压地再审一次：能过就补回
results.json 并重导出，仍不过再认账丢弃。

目标锁定 (topic, section_sid)，从备份 results.json.bak-before-rehab 取原始单元。

运行： uv run python scripts/retry_no_verdict_sop.py
"""
from __future__ import annotations

import json
from pathlib import Path

from ragkb.config import get_settings
from ragkb.llm.client import LLMClient
from ragkb.pipeline.export import load_results, export_all, _prov_from_dict
from ragkb.pipeline.gate_semantic import review_sop
from ragkb.pipeline.units import SOPUnit

_TARGETS = {
    ("【使用手册】-算法资源使用手册", "1.2.3.2"),
    ("V2.5.1-2026年2月", "1"),
    ("V2.5.1-2026年2月", "1.1"),
    ("V2.5.1-2026年2月", "1.2"),
}


def _key(unit) -> tuple[str, str]:
    src = unit.sources[0] if unit.sources else None
    return (src.topic if src else "", src.section_sid if src else "")


def main() -> None:
    settings = get_settings()
    out = settings.output_dir
    # 当前(已救回7条的)结果作为基线；load_results 已把 redaction map 载入进程，
    # 故直接解析备份 JSON 重建那4条单元，其 masked 文本能正确 restore。
    qa, sop_current = load_results(out)
    retry = _load_targets(out / "results.json.bak-before-rehab")
    print(f"锁定 {len(retry)} 条 no_verdict SOP 复审：")
    for u in retry:
        print(f"  - {_key(u)}  {u.title}")
        u.semantic_ok = None
        u.review_attempts = 0
        u.needs_review = False
        u.publication_status = "pending"

    llm = LLMClient()
    print("\n低压复审(仅审核，不重生成——内容本就未必差，先看判定)…")
    review_sop(retry, llm)   # 串行、无并发压力

    rescued = [u for u in retry if u.semantic_ok and u.publication_status == "approved"]
    still = [u for u in retry if u not in rescued]
    print(f"\n结果：救回 {len(rescued)}，仍不达标 {len(still)}")
    for u in rescued:
        print(f"  ✓ {_key(u)}  {u.title}")
    for u in still:
        print(f"  ✗ {_key(u)}  {u.title}  ← {u.semantic_reason}")

    if not rescued:
        print("\n无新增，产物不变。")
        return

    # 合并：当前 SOP 全集 + 新救回的（去重防止重复）。
    have = {_key(u) for u in sop_current}
    final_sop = sop_current + [u for u in rescued if _key(u) not in have]
    _persist(out, qa, final_sop)
    stats = export_all(qa, final_sop, out, include_paraphrases=False)
    print(f"\nresults.json 更新：SOP {len(sop_current)} → {len(final_sop)}")
    print(f"已重导出：QA {stats.qa_units}，SOP {stats.sop_files}，模块 {stats.modules}")


def load_results_from(results_path: Path, out: Path):
    """从备份 JSON 直接重建目标 SOP 单元（不动磁盘上的 live results.json）。"""
    data = json.loads(results_path.read_text("utf-8"))
    units = []
    for d in data.get("sop", []):
        u = SOPUnit(
            title=d["title"], markdown=d["markdown"],
            entry_questions=list(d.get("entry_questions", [])),
            struct_ok=d.get("struct_ok", True),
            struct_reason=d.get("struct_reason", ""),
            semantic_ok=d.get("semantic_ok"),
            semantic_reason=d.get("semantic_reason", ""),
            needs_review=d.get("needs_review", False),
            review_attempts=d.get("review_attempts", 0),
            publication_status=d.get("publication_status", "pending"),
            review_history=list(d.get("review_history", [])),
            sources=[_prov_from_dict(s) for s in d.get("sources", [])])
        units.append(u)
    return units


def _load_targets(backup: Path):
    """从备份重建、只保留锁定的 4 条 no_verdict SOP。"""
    return [u for u in load_results_from(backup, None) if _key(u) in _TARGETS]


def _persist(out: Path, qa, sop) -> None:
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
