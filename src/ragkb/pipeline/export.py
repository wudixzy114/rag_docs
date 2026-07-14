"""Export layer — produce the vector-DB ingest artifacts.

Three outputs into output/:
- `qa_pairs.csv`: STRICT two columns `Query,Answer`. One row per query key (main
  query + every paraphrase → same answer), so the recall boost materializes as
  extra rows. csv.QUOTE_ALL + utf-8-sig so CJK / commas / newlines never misalign
  and Excel opens it correctly.
- `sop/<topic>__<title>.md`: one cleaned procedure per file, with its entry
  questions embedded at the top (as front-matter-ish bullets) so the vector DB
  chunker sees the symptom phrasings alongside the procedure body.
- `metadata.jsonl`: one line per QA unit with provenance (topic/doc/heading/
  source_sha/sources). NEVER merged into the CSV — it's the traceability sidecar.

Secret scrub: internal tokens / bearer keys are regex-stripped from all exported
text before it lands in a queryable store.
"""
from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

from ragkb.pipeline.scrub import scrub
from ragkb.pipeline.units import QAUnit, SOPUnit


@dataclass
class ExportStats:
    qa_units: int = 0
    qa_rows: int = 0
    sop_files: int = 0
    needs_review: int = 0


def _safe_filename(s: str) -> str:
    s = re.sub(r"[^\w一-鿿.-]+", "_", s).strip("_")
    return s[:80] or "untitled"


def export_all(qa_units: list[QAUnit], sop_units: list[SOPUnit],
               output_dir: Path) -> ExportStats:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stats = ExportStats()

    # 1. qa_pairs.csv — strict two columns, one row per query key.
    csv_path = output_dir / "qa_pairs.csv"
    with csv_path.open("w", encoding="utf-8-sig", newline="") as f:
        w = csv.writer(f, quoting=csv.QUOTE_ALL)
        w.writerow(["Query", "Answer"])
        for u in qa_units:
            ans = scrub(u.answer)
            for key in u.query_keys():
                w.writerow([scrub(key), ans])
                stats.qa_rows += 1
            stats.qa_units += 1
            if u.needs_review:
                stats.needs_review += 1

    # 2. sop/*.md — one procedure per file, entry questions at the top.
    sop_dir = output_dir / "sop"
    sop_dir.mkdir(exist_ok=True)
    for u in sop_units:
        topic = u.sources[0].topic if u.sources else ""
        fname = f"{_safe_filename(topic)}__{_safe_filename(u.title)}.md"
        eqs = "\n".join(f"- {scrub(q)}" for q in u.entry_questions)
        header = (f"<!-- entry-questions (用户口吻入口问题，供检索路由) -->\n{eqs}\n\n"
                  if eqs else "")
        (sop_dir / fname).write_text(header + scrub(u.markdown) + "\n", "utf-8")
        stats.sop_files += 1

    # 3. metadata.jsonl — provenance sidecar (never in the CSV).
    meta_path = output_dir / "metadata.jsonl"
    with meta_path.open("w", encoding="utf-8") as f:
        for u in qa_units:
            rec = {
                "unit_id": u.unit_id,
                "query": u.query,
                "query_keys": u.query_keys(),
                "answer": scrub(u.answer),
                "needs_review": u.needs_review,
                "semantic_reason": u.semantic_reason,
                "sources": [s.to_dict() for s in u.sources],
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    return stats
