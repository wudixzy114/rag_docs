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
from ragkb.pipeline.units import Provenance, QAUnit, SOPUnit


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


def _sop_section_key(title: str) -> str:
    """The SOP filename's distinguishing part: the section NUMBER, not the title.
    The upload target caps path length, so we drop the (long) descriptive title
    and key each SOP by its dotted section id (`7.4.7`, `4.1.1`), which is unique
    within a module. Overview/FAQ pages have no number → compact `FAQ`/slug."""
    m = re.match(r"^\s*(\d+(?:\.\d+)*)", title)
    if m:
        return m.group(1)
    t = title.strip()
    if re.search(r"FAQ", t, re.I):
        return "FAQ"
    slug = re.sub(r"[^\w一-鿿]+", "_", t).strip("_")
    return slug[:8] or "untitled"


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


def _inject_entry_questions(markdown: str, questions: list[str]) -> str:
    """Fold entry-questions INTO the doc as a '常见问法' line right AFTER the first
    `#` heading, instead of a pre-heading comment block.

    Why: the vector service chunks on structure (`#`/`##`). A block placed BEFORE
    the first heading becomes its own orphan chunk — it matches question-shaped
    queries perfectly but carries NO answer, and strips the colloquial phrasing
    from the body chunk that DOES have the answer (verified in the service's
    分段预览). Injecting after the `#` keeps the phrasing in the same chunk as the
    title + intro, so a hit lands on real content."""
    if not questions:
        return markdown
    line = "**常见问法（便于检索）：** " + "｜".join(questions)
    lines = markdown.split("\n")
    for i, ln in enumerate(lines):
        if ln.lstrip().startswith("# "):
            # Insert a blank line + 问法 line right after the H1.
            rest = lines[i + 1:]
            # Drop one leading blank so we don't double up before 问法.
            if rest and rest[0].strip() == "":
                rest = rest[1:]
            return "\n".join(lines[: i + 1] + ["", line, ""] + rest)
    # No H1 (shouldn't happen — gate requires markdown to lead with one); prepend.
    return line + "\n\n" + markdown


def _write_sop(sop_dir: Path, units: list[SOPUnit]) -> int:
    sop_dir.mkdir(parents=True, exist_ok=True)
    used: set[str] = set()
    for u in units:
        topic = _module_of(u)
        key = _sop_section_key(u.title)
        stem = f"{_safe_filename(topic)}__{key}"
        # Guard: section keys are unique within a module, but defend against a
        # duplicate/missing number leaking in (append -2, -3, …) so we never
        # silently overwrite one SOP with another.
        cand, n = stem, 2
        while cand in used:
            cand = f"{stem}-{n}"
            n += 1
        used.add(cand)
        questions = [restore(q) for q in u.entry_questions]
        body = _inject_entry_questions(restore(u.markdown), questions)
        (sop_dir / f"{cand}.md").write_text(body + "\n", "utf-8")
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

    # 6. 知识库上传包.zip — the single artifact handed to the upload target.
    _write_upload_zip(output_dir)

    return stats


def _write_upload_zip(output_dir: Path) -> Path:
    """Bundle the ingest artifacts into 知识库上传包.zip.

    Two things the ad-hoc Finder/`zip` package got wrong and this fixes:
    - UTF-8 filename flag (bit 0x800) MUST be set, or CJK SOP filenames arrive
      as mojibake on `unzip`. ZipInfo.flag_bits is set explicitly (Python only
      auto-sets it for non-ascii names on some versions — we don't rely on it).
    - Never include .DS_Store / caches. We whitelist exactly what ingests:
      qa_pairs.csv, sop/, by_module/, metadata.jsonl.
    """
    import zipfile

    zip_path = output_dir / "知识库上传包.zip"
    members: list[Path] = []
    for name in ("qa_pairs.csv", "metadata.jsonl"):
        p = output_dir / name
        if p.is_file():
            members.append(p)
    for sub in ("sop", "by_module"):
        d = output_dir / sub
        if d.is_dir():
            members.extend(sorted(p for p in d.rglob("*")
                                  if p.is_file() and p.name != ".DS_Store"))

    with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for p in members:
            arcname = p.relative_to(output_dir).as_posix()
            info = zipfile.ZipInfo(arcname)
            info.compress_type = zipfile.ZIP_DEFLATED
            info.flag_bits |= 0x800          # UTF-8 filename flag
            info.external_attr = 0o644 << 16
            zf.writestr(info, p.read_bytes())
    return zip_path


def _prov_from_dict(d: dict) -> Provenance:
    return Provenance(
        topic=d.get("topic", ""), doc_title=d.get("doc_title", ""),
        heading_path=d.get("heading_path", ""), section_sid=d.get("section_sid", ""),
        source_sha=d.get("source_sha", ""), image_refs=list(d.get("image_refs", [])))


def load_results(output_dir: Path) -> tuple[list[QAUnit], list[SOPUnit]]:
    """Rehydrate consolidated QA/SOP units from results.json so `ragkb export`
    can re-run outside the producing process. Also reloads redaction_map.json
    into the process-global redactor — results.json stores MASKED text, so
    without the map `restore()` would leave placeholders in the output.

    Units here are already post-consolidation (aggregated + reviewed + deduped),
    so export writes them verbatim; no re-gating."""
    output_dir = Path(output_dir)
    rmap = output_dir / "redaction_map.json"
    if rmap.is_file():
        from ragkb.pipeline import scrub
        audit = json.loads(rmap.read_text("utf-8"))     # token → real value
        for tok, val in audit.items():
            scrub._DEFAULT._to_value[tok] = val
            scrub._DEFAULT._to_token[val] = tok

    data = json.loads((output_dir / "results.json").read_text("utf-8"))
    qa = [QAUnit(query=d["query"], answer=d["answer"],
                 paraphrases=list(d.get("paraphrases", [])),
                 needs_review=d.get("needs_review", False),
                 semantic_reason=d.get("semantic_reason", ""),
                 sources=[_prov_from_dict(s) for s in d.get("sources", [])])
          for d in data.get("qa", [])]
    sop = [SOPUnit(title=d["title"], markdown=d["markdown"],
                   entry_questions=list(d.get("entry_questions", [])),
                   struct_ok=d.get("struct_ok", True),
                   sources=[_prov_from_dict(s) for s in d.get("sources", [])])
           for d in data.get("sop", [])]
    return qa, sop
