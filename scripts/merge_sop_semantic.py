"""语义合并 SOP：把 121 篇碎片按【语义主题】合并成 ~24 个文件，用于上传。

这是数据清洗，不受原文件目录束缚——按内容语义把同主题的碎片合到一起：

- 大部头结构化文档(使用手册、产品介绍)：按【顶层大章】合并
  (如"2. 算力资源管理"下的 2.1/2.2/2.3… 合成一篇)。
- 其余模块(各版本发布、产品动态、基本概念、操作指南)：整模块合成一篇。

质量优先且与导出口径完全一致：
- 复用 export 的 load_results(含 restore 脱敏还原) + _inject_entry_questions，
  所以每节正文、常见问法与单篇导出逐字相同，只是拼接在一起。
- 每个合并文件 `# <主题>` 作 H1，各节内容整体下沉一级(H1→H2…)保持层级合法。
- 节按 section_sid 数值排序，连贯有序。

产出：output/upload_sop/<安全文件名>.md。QA 侧 qa_pairs.csv 本就单文件，不动。

运行： uv run python scripts/merge_sop_semantic.py
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from pathlib import Path

from ragkb.config import get_settings
from ragkb.pipeline.export import load_results, _inject_entry_questions
from ragkb.pipeline.scrub import restore

_STRUCTURED = {"【使用手册】-算法资源使用手册", "产品介绍和功能概述"}


def _sid_key(sid: str) -> tuple:
    return tuple(int(p) if p.isdigit() else 9999 for p in (sid or "").split("."))


def _demote(md: str) -> str:
    lines = []
    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})(\s+.*)$", line)
        lines.append("#" * min(6, len(m.group(1)) + 1) + m.group(2) if m else line)
    return "\n".join(lines)


def _safe(name: str) -> str:
    return re.sub(r"[^\w一-鿿.-]+", "_", name).strip("_")[:100] or "untitled"


def _group_of(unit) -> tuple[str, str]:
    src = unit.sources[0] if unit.sources else None
    module = src.topic if src else "_unknown"
    if module in _STRUCTURED:
        parts = [p.strip() for p in (src.heading_path if src else "").split("/")]
        top = parts[1] if len(parts) >= 2 else unit.title
        return (f"{module}::{top}", f"{module} — {top}")
    return (module, module)


def main() -> None:
    settings = get_settings()
    out = settings.output_dir
    # load_results 同时载入 redaction map，使 restore() 能还原真实值(与导出一致)。
    _, sop = load_results(out)
    sop = [u for u in sop
           if u.publication_status != "failed_review" and u.semantic_ok is not False
           and u.struct_ok]

    groups: dict[str, dict] = {}
    for u in sop:
        key, label = _group_of(u)
        groups.setdefault(key, {"label": label, "units": []})["units"].append(u)

    dest = out / "upload_sop"
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    print(f"语义合并 {len(sop)} 篇 SOP → {len(groups)} 个文件\n")
    used: set[str] = set()
    total_chars = 0
    for key in sorted(groups):
        g = groups[key]
        units = sorted(g["units"],
                       key=lambda u: _sid_key(u.sources[0].section_sid if u.sources else ""))
        parts = [f"# {g['label']}\n"]
        for u in units:
            # 与 export._write_sop 完全相同的渲染：restore 还原 + 注入常见问法。
            questions = [restore(q) for q in u.entry_questions]
            body = _inject_entry_questions(restore(u.markdown), questions)
            parts.append(_demote(body.strip()))
            parts.append("")
        merged = "\n".join(parts).strip() + "\n"
        fname = _safe(g["label"].replace(" — ", "__"))
        cand, n = fname, 2
        while cand in used:
            cand = f"{fname}-{n}"; n += 1
        used.add(cand)
        (dest / f"{cand}.md").write_text(merged, "utf-8")
        total_chars += len(merged)
        print(f"  {len(units):2d}篇 → {cand}.md ({len(merged)}字符)")

    print(f"\n完成：{len(sop)} → {len(groups)} 个文件，输出 {dest}")
    print(f"合计字符：{total_chars}")


if __name__ == "__main__":
    main()
