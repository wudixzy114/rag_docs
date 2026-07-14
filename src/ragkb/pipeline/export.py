"""Export layer — produce the vector-DB ingest artifacts.

Outputs into output/:
- `qa_pairs.csv`: three columns `Query,Answer,Module`. One row per query key (main
  query + every paraphrase → same answer), so the recall boost materializes as
  extra rows. `Module` is a SOFT label — the vector DB indexes globally and may
  use it to filter/boost when the caller's module is known, but retrieval is NOT
  hard-partitioned (a symptom query must reach every module). QUOTE_ALL +
  utf-8-sig so CJK / commas / newlines never misalign.
- `by_module/<module>/qa_pairs.csv` + `by_module/<module>/sop/`: the same content
  partitioned per module, for callers that ingest one module at a time. Dedup is
  module-scoped upstream, so each module's file is self-contained.
- `sop/<module>__<title>.md`: global SOP dir (all procedures).
- `metadata.jsonl`: provenance sidecar (never in the CSV).
- `redaction_map.json`: token→real-value audit trail for the reversible masking.

Reversible masking: text was masked (MAC/IP/phone/keys → placeholders) BEFORE
being sent to the gateway. On export we RESTORE the real values — the KB is
internal, so everything is visible. `restore()` is the single choke point.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ragkb.pipeline.scrub import mapping, restore
from ragkb.pipeline.units import QAUnit, SOPUnit


@dataclass
class ExportStats:
    qa_units: int = 0
    qa_rows: int = 0
    sop_files: int = 0
    needs_review: int = 0
    modules: int = 0


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w一-鿿.-]+", "_", s).strip("_")
    return s[:80] or "untitled"


def _module_of(u) -> str:
    return u.sources[0].topic if u.sources else "_unknown"


def _write_qa_csv(path: Path, units: list[QAUnit]) -> int:
    """Write a Query,Answer,Module CSV. Returns row count. Values are RESTORED
    (real MAC/IP/etc.) since the KB is internal."""
    rows = 0
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["Query", "Answer", "Module"])
        for u in units:
            ans = restore(u.answer)
            mod = _module_of(u)
            for key in u.query_keys():
                w.writerow([restore(key), ans, mod])
                rows += 1
    return rows


def _write_sop(sop_dir: Path, units: list[SOPUnit]) -> int:
    sop_dir.mkdir(parents=True, exist_ok=True)
    for u in units:
        topic = _module_of(u)
        fname = f"{_safe_filename(topic)}__{_safe_filename(u.title)}.md"
        eqs = "\n".join(f"- {restore(q)}" for q in u.entry_questions)
        header = (f"<!-- entry-questions (用户口吻入口问题，供检索路由) -->\n{eqs}\n\n"
                  if eqs else "")
        (sop_dir / fname).write_text(header + restore(u.markdown) + "\n", "utf-8")
    return len(units)


def export_all(qa_units: list[QAUnit], sop_units: list[SOPUnit],
               output_dir: Path) -> ExportStats:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()

    # 1. Global qa_pairs.csv (all modules, with Module column).
    stats.qa_rows = _write_qa_csv(output_dir / "qa_pairs.csv", qa_units)
    stats.qa_units = len(qa_units)
    stats.needs_review = sum(1 for u in qa_units if u.needs_review)

    # 2. Global sop/ dir.
    stats.sop_files = _write_sop(output_dir / "sop", sop_units)

    # 3. Per-module partition: by_module/<module>/{qa_pairs.csv, sop/}.
    modules: dict[str, dict] = {}
    for u in qa_units:
        modules.setdefault(_module_of(u), {"qa": [], "sop": []})["qa"].append(u)
    for u in sop_units:
        modules.setdefault(_module_of(u), {"qa": [], "sop": []})["sop"].append(u)
    by_module = output_dir / "by_module"
    for mod, bucket in modules.items():
        mdir = by_module / _safe_filename(mod)
        if bucket["qa"]:
            _write_qa_csv(mdir / "qa_pairs.csv", bucket["qa"])
        if bucket["sop"]:
            _write_sop(mdir / "sop", bucket["sop"])
    stats.modules = len(modules)

    # 4. metadata.jsonl — provenance sidecar (restored values, module label).
    with (output_dir / "metadata.jsonl").open("w", encoding="utf-8") as f:
        for u in qa_units:
            rec = {
                "unit_id": u.unit_id,
                "module": _module_of(u),
                "query": restore(u.query),
                "query_keys": [restore(k) for k in u.query_keys()],
                "answer": restore(u.answer),
                "needs_review": u.needs_review,
                "semantic_reason": u.semantic_reason,
                "sources": [s.to_dict() for s in u.sources],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # 5. redaction_map.json — audit trail of every masked value (token → real).
    audit = mapping()
    if audit:
        (output_dir / "redaction_map.json").write_text(
            json.dumps(audit, ensure_ascii=False, indent=2), "utf-8")

    return stats
