"""SOP 语义合并 —— 把碎片化的 SOP 单元按【语义主题】合并成少量文件，用于上传。

背景：抽取产出的 SOP 是「每 section 一篇」，全量下会有上千个碎文件，上传数量爆炸。
这一步是数据清洗（不受原文件目录结构束缚）：按内容语义把同主题的碎片合到一起。

合并粒度（自适应，不依赖硬编码模块名，以适配任意来源）：
- 「大部头结构化文档」：section 多且标题层级深的文档，按【顶层大章】拆分合并
  （如"2. 算力资源管理"下的 2.1/2.2/2.3… 合成一篇），控制单文件大小又语义清晰。
- 其余模块（版本发布、产品动态、概念、操作指南）：整模块合成一篇。

质量优先，且与导出口径完全一致：
- 复用 export 的 load_results（含 restore 脱敏还原）+ _inject_entry_questions，
  每节正文、常见问法与单篇导出逐字相同，只是拼接在一起。
- 每个合并文件以 `# <主题>` 作 H1，各节内容整体下沉一级(H1→H2…)保持层级合法。
- 节按 section_sid 数值排序，连贯有序。

产出到 <output>/upload_sop/<安全文件名>.md。QA 侧是单文件 CSV，无需合并。
"""
from __future__ import annotations

import re
import shutil
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from ragkb.pipeline.export import load_results, _inject_entry_questions
from ragkb.pipeline.scrub import restore
from ragkb.pipeline.units import SOPUnit

# 一个模块 section 数达到此阈值、且存在多个顶层大章时，按大章拆分而非整篇合并。
# 经验值：低于它整篇合更连贯；高于它整篇会过大，按大章更利于向量库分段与阅读。
_SPLIT_SECTION_THRESHOLD = 6


@dataclass
class MergeStats:
    sop_units: int = 0
    files: int = 0
    total_chars: int = 0
    modules: int = 0


def _sid_key(sid: str) -> tuple:
    """把 '1.2.3' 变成 (1,2,3) 数值排序键；非数字段回退大值排到末尾。"""
    return tuple(int(p) if p.isdigit() else 9999 for p in (sid or "").split("."))


def _demote(md: str) -> str:
    """整篇标题下沉一级(#→##…)，最多到 ######，作为合并文件的子章节。"""
    out = []
    for line in md.split("\n"):
        m = re.match(r"^(#{1,6})(\s+.*)$", line)
        out.append("#" * min(6, len(m.group(1)) + 1) + m.group(2) if m else line)
    return "\n".join(out)


def _safe(name: str) -> str:
    return re.sub(r"[^\w一-鿿.-]+", "_", name).strip("_")[:100] or "untitled"


def _module_of(u: SOPUnit) -> str:
    return u.sources[0].topic if u.sources else "_unknown"


def _top_chapter(u: SOPUnit) -> str:
    """heading_path 的顶层大章（模块名之后的第一级），无则回退到标题。"""
    src = u.sources[0] if u.sources else None
    parts = [p.strip() for p in (src.heading_path if src else "").split("/") if p.strip()]
    return parts[1] if len(parts) >= 2 else u.title


def _plan_groups(sop: list[SOPUnit]) -> dict[str, dict]:
    """把 SOP 单元分组为最终合并文件。返回 {group_key: {label, units}}。

    自适应粒度，区分两类文档：
    - 「层级文档」（如使用手册：1./2./3. 大章，各大章下有多个子节）→ 按顶层大章拆，
      每大章合一篇。判据是「存在某个顶层大章聚集了多个 section」——即大章有厚度。
    - 「扁平文档」（如产品动态：一堆平级日期条目；基本概念：一堆平级概念）→ 整模块合一篇。
      这类每个 section 的顶层大章几乎各不相同（大章无厚度），不该按大章拆散。

    只有当「最厚的顶层大章 section 数 ≥ 2」且「模块 section 数超阈值」时才拆分。"""
    by_mod: dict[str, list[SOPUnit]] = defaultdict(list)
    for u in sop:
        by_mod[_module_of(u)].append(u)

    groups: dict[str, dict] = {}
    for module, units in by_mod.items():
        chapter_sizes: dict[str, int] = defaultdict(int)
        for u in units:
            chapter_sizes[_top_chapter(u)] += 1
        thickest = max(chapter_sizes.values()) if chapter_sizes else 0
        # 层级文档：模块够大 且 至少一个大章聚了 ≥2 个子节（有真正的层级结构）。
        if len(units) > _SPLIT_SECTION_THRESHOLD and thickest >= 2:
            for u in units:
                ch = _top_chapter(u)
                key = f"{module}::{ch}"
                groups.setdefault(key, {"label": f"{module} — {ch}", "units": []})
                groups[key]["units"].append(u)
        else:
            groups.setdefault(module, {"label": module, "units": []})
            groups[module]["units"].extend(units)
    return groups


def merge_sop(output_dir: Path, subdir: str = "upload_sop") -> MergeStats:
    """从 output_dir/results.json 读取可发布 SOP，语义合并后写入 output_dir/<subdir>/。

    只合并可发布单元（approved 且结构有效），与 export 的发布口径一致。"""
    output_dir = Path(output_dir)
    # load_results 同时载入 redaction map，使 restore() 还原真实值（与导出一致）。
    _, sop = load_results(output_dir)
    sop = [u for u in sop
           if u.publication_status != "failed_review"
           and u.semantic_ok is not False and u.struct_ok]

    groups = _plan_groups(sop)

    dest = output_dir / subdir
    if dest.exists():
        shutil.rmtree(dest)
    dest.mkdir(parents=True, exist_ok=True)

    stats = MergeStats(sop_units=len(sop), modules=len({_module_of(u) for u in sop}))
    used: set[str] = set()
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
        # 文件名去重（防同名大章跨模块碰撞）。
        stem = _safe(g["label"].replace(" — ", "__"))
        cand, n = stem, 2
        while cand in used:
            cand, n = f"{stem}-{n}", n + 1
        used.add(cand)
        (dest / f"{cand}.md").write_text(merged, "utf-8")
        stats.files += 1
        stats.total_chars += len(merged)
    return stats
