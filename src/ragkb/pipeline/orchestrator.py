"""Orchestrator — parallel, observable, idempotent pipeline driver.

Runs the whole flow and is the single entry point the CLI and the dashboard call:

  discover topics
    → per-doc (ThreadPoolExecutor, bounded by settings.max_workers):
        parse → vision-read images → classify sections → extract QA/SOP
              → Layer1 struct gate → Layer2 semantic gate → paraphrase
    → cross-doc aggregate (by topic) → global dedup
    → store results in memory + on disk (results.json), ready for export/pin

Observability: every state change publishes an Event (picked up by the dashboard
SSE stream). Idempotency: the manifest skips pinned/unchanged docs. Retry: run(
only=[topics]) re-processes just those; a pinned doc is never overwritten.

Concurrency note: the vendored LLMClient is thread-safe (thread-local active
model), so one shared client is used across workers. The single global ceiling is
settings.max_workers (extraction pool) — the semantic gate's internal pool is
small and runs after extraction, so they don't compound catastrophically; keep
max_workers modest per the gateway rate limit.
"""
from __future__ import annotations

import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path

from ragkb.config import Settings, get_settings
from ragkb.llm.client import LLMClient
from ragkb.parse.markdown import parse_document
from ragkb.parse.model import Document
from ragkb.pipeline.aggregate import aggregate_by_topic
from ragkb.pipeline.classify import classify_sections
from ragkb.pipeline.dedup import dedup_qa
from ragkb.pipeline.events import EventBus
from ragkb.pipeline.extract import extract_qa, extract_sop
from ragkb.pipeline.gate_semantic import review_qa
from ragkb.pipeline.gate_struct import gate_qa, gate_sop
from ragkb.pipeline.paraphrase import add_paraphrases
from ragkb.pipeline.units import QAUnit, SOPUnit
from ragkb.pipeline.vision import vision_read_image
from ragkb.store.cache import Cache
from ragkb.store.manifest import DocState, Manifest

log = logging.getLogger(__name__)


@dataclass
class DocResult:
    topic: str
    qa: list[QAUnit] = field(default_factory=list)
    sop: list[SOPUnit] = field(default_factory=list)
    error: str = ""


def discover_topics(input_dir: Path) -> list[Path]:
    """Each subfolder containing 原始文档.md is one topic/document."""
    out = []
    if not input_dir.is_dir():
        return out
    # The material nests under a single top folder (知识库配对素材); descend into it.
    roots = [input_dir]
    for child in sorted(input_dir.iterdir()):
        if child.is_dir():
            roots.append(child)
    seen = set()
    for root in roots:
        for md in sorted(root.glob("*/原始文档.md")):
            if md.parent not in seen:
                seen.add(md.parent)
                out.append(md)
    return out


class Orchestrator:
    def __init__(self, settings: Settings | None = None,
                 bus: EventBus | None = None,
                 llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        self.bus = bus or EventBus()
        self.llm = llm or LLMClient()
        self.cache = Cache(self.settings.cache_dir)
        self.manifest = Manifest(self.settings.output_dir / "manifest.json")
        self._results: dict[str, DocResult] = {}
        self._lock = threading.Lock()
        self._running = False

    # ---- per-document pipeline -------------------------------------------
    def _process_doc(self, md_path: Path) -> DocResult:
        topic = md_path.parent.name
        self.bus.publish("doc_status", topic, status="running")
        self.manifest.update(topic, status="running", error="")
        try:
            doc = parse_document(md_path)
            self.manifest.update(topic, source_sha=doc.source_sha)
            # 1. Vision-read every image (replaces weak inline OCR).
            imgs = doc.all_images()
            for k, im in enumerate(imgs):
                vision_read_image(im, self.llm, self.cache)
                self.bus.publish("doc_progress", topic, stage="vision",
                                 done=k + 1, total=len(imgs))
            # 2. Classify sections.
            secs = [s for s in doc.iter_sections() if s.body.strip() or s.images]
            labels = classify_sections(secs, self.llm)
            # 3. Extract per section by label.
            qa: list[QAUnit] = []
            sop: list[SOPUnit] = []
            for k, s in enumerate(secs):
                label = labels.get(s.sid, "qa")
                if label == "qa":
                    qa.extend(extract_qa(doc, s, self.llm, cache=self.cache))
                elif label == "sop":
                    u = extract_sop(doc, s, self.llm, cache=self.cache)
                    if u:
                        sop.append(u)
                self.bus.publish("doc_progress", topic, stage="extract",
                                 done=k + 1, total=len(secs))
            # 4. Layer 1 structural gate (retry truncated once with more tokens).
            for u in qa:
                gate_qa(u)
            for u in sop:
                gate_sop(u)
            qa = self._retry_truncated_qa(doc, secs, labels, qa)
            # 5. Paraphrase (recall boost) — only for struct-ok units.
            for u in qa:
                if u.struct_ok:
                    add_paraphrases(u, self.llm, self.cache)
            res = DocResult(topic=topic, qa=qa, sop=sop)
            self._publish_units(topic, qa, sop)
            return res
        except Exception as exc:                      # noqa: BLE001 - isolate one doc's failure
            log.exception("doc %s failed", topic)
            self.bus.publish("error", topic, message=str(exc))
            self.manifest.update(topic, status="failed", error=str(exc))
            return DocResult(topic=topic, error=str(exc))

    def _retry_truncated_qa(self, doc, secs, labels, qa):
        """Re-extract sections that produced a truncated unit, with a bigger
        budget. Prevents silently shipping a half answer (user's forbidden
        failure mode)."""
        bad_sids = {p.section_sid for u in qa if u.truncated for p in u.sources}
        if not bad_sids:
            return qa
        by_sid = {s.sid: s for s in secs}
        good = [u for u in qa if not u.truncated]
        for sid in bad_sids:
            s = by_sid.get(sid)
            if not s:
                continue
            retried = extract_qa(doc, s, self.llm)  # extract_qa uses a 4096 budget
            for u in retried:
                gate_qa(u)
            good.extend(retried)
            self.bus.publish("log", doc.topic,
                             message=f"retried truncated section {sid}")
        return good

    def _publish_units(self, topic, qa, sop):
        for u in qa:
            self.bus.publish("unit", topic, unit_kind="qa", query=u.query,
                             answer=u.answer, struct_ok=u.struct_ok,
                             truncated=u.truncated)
        for u in sop:
            self.bus.publish("unit", topic, unit_kind="sop", title=u.title,
                             entry_questions=u.entry_questions,
                             struct_ok=u.struct_ok)

    # ---- run --------------------------------------------------------------
    def run(self, only: list[str] | None = None, force: bool = False) -> dict[str, DocResult]:
        """Process all (or `only`) topics in parallel, then aggregate + review +
        dedup globally. `force` ignores the idempotency skip (but never touches
        pinned docs)."""
        self._running = True
        self.bus.publish("run_status", status="started")
        md_paths = discover_topics(self.settings.input_dir)
        if only:
            md_paths = [p for p in md_paths if p.parent.name in set(only)]

        # Seed manifest entries and decide skips.
        todo: list[Path] = []
        for p in md_paths:
            topic = p.parent.name
            sha = _sha_of(p)
            if not self.manifest.get(topic):
                self.manifest.upsert(DocState(topic=topic, source_sha=sha))
            st = self.manifest.get(topic)
            if st and st.pinned:
                self.bus.publish("doc_status", topic, status="pinned")
                continue
            if not force and self.manifest.should_skip(topic, sha):
                self.bus.publish("doc_status", topic, status="skipped")
                continue
            todo.append(p)

        # Parallel per-doc processing.
        workers = min(self.settings.max_workers, max(1, len(todo)))
        if todo:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futs = {pool.submit(self._process_doc, p): p for p in todo}
                for fut in as_completed(futs):
                    res = fut.result()
                    with self._lock:
                        self._results[res.topic] = res

        # Global consolidation across everything we have (fresh + previously done
        # non-pinned results in memory this run).
        self._consolidate()
        self._persist_results()
        self.bus.publish("run_status", status="done")
        self._running = False
        return dict(self._results)

    def _consolidate(self) -> None:
        """Cross-file aggregate + semantic review + dedup over the union of all QA."""
        all_qa: list[QAUnit] = []
        for res in self._results.values():
            all_qa.extend([u for u in res.qa if u.struct_ok])
        if not all_qa:
            return
        # Cross-file aggregate (by topic), then one global semantic review, then dedup.
        aggregated = aggregate_by_topic(all_qa)
        self.bus.publish("log", message=f"aggregated {len(all_qa)}→{len(aggregated)} QA; reviewing…")
        review_qa(aggregated, self.llm, policy=self.settings.semantic_gate_policy)
        survivors = [u for u in aggregated if u.semantic_ok]
        deduped = dedup_qa(survivors)
        self.bus.publish("log",
                         message=f"review kept {len(survivors)}/{len(aggregated)}; dedup→{len(deduped)}")
        # Redistribute consolidated QA back to their primary topic for per-doc stats.
        by_topic: dict[str, list[QAUnit]] = {}
        for u in deduped:
            topic = u.sources[0].topic if u.sources else ""
            by_topic.setdefault(topic, []).append(u)
        for topic, res in self._results.items():
            res.qa = by_topic.get(topic, [])
            self.manifest.update(
                topic, status="done",
                extracted=len(res.qa), passed=len(res.qa),
                needs_review=sum(1 for u in res.qa if u.needs_review),
                sop_count=len(res.sop))
        # Stash the fully consolidated set for export.
        self._consolidated_qa = deduped

    def _persist_results(self) -> None:
        out = self.settings.output_dir / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"qa": [], "sop": []}
        seen = set()
        for res in self._results.values():
            for u in res.qa:
                if u.unit_id in seen:
                    continue
                seen.add(u.unit_id)
                payload["qa"].append({
                    "query": u.query, "answer": u.answer,
                    "paraphrases": u.paraphrases,
                    "needs_review": u.needs_review,
                    "semantic_reason": u.semantic_reason,
                    "sources": [s.to_dict() for s in u.sources]})
            for u in res.sop:
                payload["sop"].append({
                    "title": u.title, "markdown": u.markdown,
                    "entry_questions": u.entry_questions,
                    "struct_ok": u.struct_ok,
                    "sources": [s.to_dict() for s in u.sources]})
        tmp = out.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
        tmp.replace(out)

    def results(self) -> dict[str, DocResult]:
        with self._lock:
            return dict(self._results)

    def export(self):
        """Write qa_pairs.csv + sop/*.md + metadata.jsonl from the consolidated
        results of the last run. Returns ExportStats."""
        from ragkb.pipeline.export import export_all
        qa = getattr(self, "_consolidated_qa", None)
        if qa is None:
            qa = []
            seen = set()
            for res in self._results.values():
                for u in res.qa:
                    if u.unit_id not in seen:
                        seen.add(u.unit_id)
                        qa.append(u)
        sop = [u for res in self._results.values() for u in res.sop if u.struct_ok]
        stats = export_all(qa, sop, self.settings.output_dir)
        self.bus.publish("run_status", status="exported",
                         qa_units=stats.qa_units, qa_rows=stats.qa_rows,
                         sop_files=stats.sop_files, needs_review=stats.needs_review)
        return stats


def _sha_of(path: Path) -> str:
    import hashlib
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()
