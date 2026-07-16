"""Orchestrator — parallel, observable, idempotent pipeline driver.

Runs the whole flow and is the single entry point the CLI and the dashboard call:

  discover topics
    → per-doc (ThreadPoolExecutor, bounded by settings.max_workers):
        parse → vision-read images → classify sections → extract QA/SOP
              → Layer1 struct gate → Layer2 semantic gate → one bounded regeneration
    → cross-doc aggregate (by topic) → global dedup
    → store results in memory + on disk (results.json), ready for export/pin

Observability: every state change publishes an Event (picked up by the dashboard
SSE stream). Idempotency: the manifest skips pinned/unchanged docs. Retry: run(
only=[topics]) re-processes just those; a pinned doc is never overwritten.

Concurrency note: documents are the only executor boundary. One thread-safe
client applies a second global semaphore to every network call, so document
parallelism, retries and later stages cannot multiply into nested request bursts.
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
from ragkb.parse.source import SUPPORTED_EXTENSIONS, source_bundle_sha
from ragkb.pipeline.aggregate import aggregate_by_topic
from ragkb.pipeline.classify import classify_sections
from ragkb.pipeline.dedup import dedup_qa
from ragkb.pipeline.events import EventBus
from ragkb.pipeline.extract import extract_qa, extract_qa_sections, extract_sop
from ragkb.pipeline.gate_struct import gate_qa, gate_sop
from ragkb.pipeline.paraphrase import add_paraphrases_batch
from ragkb.pipeline.regenerate import (
    review_sop_with_regeneration,
    review_with_regeneration,
)
from ragkb.pipeline.sections import split_oversize_sections
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
    """Recursively discover every supported source file, independent of naming."""
    if not input_dir.is_dir():
        return []
    return sorted(p for p in input_dir.rglob("*")
                  if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
                  and not any(part.startswith(".") for part in p.relative_to(input_dir).parts))


def _topic_for(path: Path, input_dir: Path) -> str:
    """Stable human-readable document key, compatible with 原始文档.md layouts."""
    if path.stem.lower() in {"原始文档", "index"} and path.parent != input_dir:
        return path.parent.name
    parent = path.parent.name if path.parent != input_dir else ""
    return f"{parent}__{path.stem}" if parent else path.stem


class Orchestrator:
    def __init__(self, settings: Settings | None = None,
                 bus: EventBus | None = None,
                 llm: LLMClient | None = None) -> None:
        self.settings = settings or get_settings()
        # Default bus journals to output/events.jsonl so a CLI run and the
        # dashboard server (separate processes) share one cross-process event
        # stream. A caller that passes its own bus opts into its own policy.
        self.bus = bus or EventBus(journal_path=self.settings.output_dir / "events.jsonl")
        self.llm = llm or LLMClient()
        self.cache = Cache(self.settings.cache_dir)
        self.manifest = Manifest(self.settings.output_dir / "manifest.json")
        self._results: dict[str, DocResult] = {}
        self._lock = threading.Lock()
        self._running = False
        self._failed_this_run: set[str] = set()
        self._load_previous_results()

    def _load_previous_results(self) -> None:
        """Hydrate the last successful snapshot for incremental/crash-safe runs."""
        results_file = self.settings.output_dir / "results.json"
        if not results_file.is_file():
            return
        try:
            from ragkb.pipeline.export import load_results
            qa, sop = load_results(self.settings.output_dir)
        except (OSError, ValueError, KeyError, json.JSONDecodeError) as exc:
            log.warning("cannot load previous results: %s", exc)
            return
        for unit in qa:
            if not self.settings.enable_paraphrases:
                unit.paraphrases = []
            topic = unit.sources[0].topic if unit.sources else ""
            self._results.setdefault(topic, DocResult(topic=topic)).qa.append(unit)
        for unit in sop:
            topic = unit.sources[0].topic if unit.sources else ""
            self._results.setdefault(topic, DocResult(topic=topic)).sop.append(unit)
        self._consolidated_qa = [u for u in qa if u.semantic_ok]

    # ---- per-document pipeline -------------------------------------------
    def _process_doc(self, md_path: Path, topic: str | None = None) -> DocResult:
        topic = topic or _topic_for(md_path, self.settings.input_dir)
        self.bus.publish("doc_status", topic, status="running")
        self.manifest.update(topic, status="running", error="")
        try:
            self.bus.publish("stage", topic, stage="parse", status="running")
            doc = parse_document(md_path, topic=topic)
            self.manifest.update(topic, source_sha=doc.source_sha)
            self.bus.publish("stage", topic, stage="parse", status="done",
                             detail=f"{sum(1 for _ in doc.iter_sections())} sections")
            # 1. Vision-read every image (replaces weak inline OCR).
            imgs = doc.all_images()
            self.bus.publish("stage", topic, stage="vision", status="running",
                             done=0, total=len(imgs))
            for k, im in enumerate(imgs):
                vision_read_image(im, self.llm, self.cache)
                self.bus.publish("doc_progress", topic, stage="vision",
                                 done=k + 1, total=len(imgs))
                self.bus.publish("stage", topic, stage="vision", status="running",
                                 done=k + 1, total=len(imgs))
            self.bus.publish("stage", topic, stage="vision", status="done",
                             done=len(imgs), total=len(imgs))
            # 2. Classify sections.
            secs = split_oversize_sections(
                [s for s in doc.iter_sections() if s.body.strip() or s.images])
            self.bus.publish("stage", topic, stage="classify", status="running",
                             done=0, total=len(secs))
            labels = classify_sections(secs, self.llm)
            self.bus.publish("stage", topic, stage="classify", status="done",
                             done=len(secs), total=len(secs))
            # 3. Extract with size-aware batches. Documents are the concurrency
            #    boundary, so there is no nested pool multiplying gateway pressure.
            sop: list[SOPUnit] = []
            qa_secs = [s for s in secs if labels.get(s.sid, "qa") == "qa"]
            sop_secs = [s for s in secs if labels.get(s.sid, "qa") == "sop"]
            total = len(qa_secs) + len(sop_secs)
            self.bus.publish("stage", topic, stage="extract", status="running",
                             done=0, total=total)
            qa = extract_qa_sections(doc, qa_secs, self.llm, cache=self.cache)
            self.bus.publish("doc_progress", topic, stage="extract",
                             done=len(qa_secs), total=total)
            for done, section in enumerate(sop_secs, start=len(qa_secs) + 1):
                unit = extract_sop(doc, section, self.llm, cache=self.cache)
                if unit:
                    sop.append(unit)
                self.bus.publish("doc_progress", topic, stage="extract",
                                 done=done, total=total)
            self.bus.publish("stage", topic, stage="extract", status="done",
                             done=total, total=total,
                             detail=f"{len(qa)} QA / {len(sop)} SOP")
            # 4. Layer 1 structural gate (retry truncated once with more tokens).
            self.bus.publish("stage", topic, stage="validate", status="running")
            for u in qa:
                gate_qa(u)
            for u in sop:
                gate_sop(u)
            qa = self._retry_truncated_qa(doc, secs, labels, qa)
            sop = self._retry_truncated_sop(doc, secs, sop)
            # A section classified 'qa' that yields no structurally valid QA is
            # almost always announcement/说明 content mislabeled by the classifier
            # (release notes, concept intros). Rather than fail the whole doc on the
            # coverage invariant, retry it as SOP. Emits a visible 'fallback' event
            # per section so the reclassification is never silent.
            qa, sop = self._fallback_qa_to_sop(doc, secs, labels, qa, sop)
            self._assert_extraction_coverage(secs, labels, qa, sop)
            self.bus.publish("stage", topic, stage="validate", status="done",
                             detail=(f"{sum(u.struct_ok for u in qa)} QA / "
                                     f"{sum(u.struct_ok for u in sop)} SOP valid"))
            # Optional experiment only; the production path remains source-faithful.
            ok_units = [u for u in qa if u.struct_ok]
            if self.settings.enable_paraphrases and ok_units:
                add_paraphrases_batch(ok_units, self.llm, self.cache)
            res = DocResult(topic=topic, qa=qa, sop=sop)
            self._publish_units(topic, qa, sop)
            return res
        except Exception as exc:                      # noqa: BLE001 - isolate one doc's failure
            log.exception("doc %s failed", topic)
            self.bus.publish("error", topic, message=str(exc))
            self.manifest.update(topic, status="failed", error=str(exc))
            self.bus.publish("stage", topic, stage="failed", status="failed",
                             detail=str(exc))
            with self._lock:
                self._failed_this_run.add(topic)
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
            retried = extract_qa(doc, s, self.llm, cache=self.cache, max_tokens=8192)
            for u in retried:
                gate_qa(u)
            good.extend(retried)
            self.bus.publish("log", doc.topic,
                             message=f"retried truncated section {sid}")
        return good

    def _retry_truncated_sop(self, doc, secs, sop):
        bad_sids = {p.section_sid for u in sop if u.truncated for p in u.sources}
        if not bad_sids:
            return sop
        by_sid = {s.sid: s for s in secs}
        good = [u for u in sop if not u.truncated]
        for sid in bad_sids:
            section = by_sid.get(sid)
            if not section:
                continue
            retried = extract_sop(doc, section, self.llm, cache=self.cache,
                                  max_tokens=12288)
            if retried:
                gate_sop(retried)
                good.append(retried)
            self.bus.publish("log", doc.topic,
                             message=f"retried truncated SOP section {sid}")
        return good

    def _fallback_qa_to_sop(self, doc, secs, labels, qa, sop):
        """Retry 'qa' sections that produced no valid QA as SOP extraction.

        The classifier occasionally labels announcement/说明 content (release
        notes, concept pages) as 'qa'; the extractor then honestly returns [],
        which would trip the coverage invariant and fail the whole document. This
        catches those sections, re-extracts each as an SOP, and — on success —
        moves it into the SOP set and rewrites its label so coverage passes.

        Every reclassification publishes a 'fallback' event (topic + section +
        title + outcome) so the operator sees exactly what was reclassified and
        why — never a silent behavior change.
        """
        qa_ok_sids = {p.section_sid for u in qa if u.struct_ok for p in u.sources}
        by_sid = {s.sid: s for s in secs}
        gap_sids = [s.sid for s in secs
                    if labels.get(s.sid, "qa") == "qa" and s.sid not in qa_ok_sids]
        if not gap_sids:
            return qa, sop
        for sid in gap_sids:
            section = by_sid.get(sid)
            if section is None:
                continue
            self.bus.publish(
                "fallback", doc.topic, section=sid, title=section.title,
                heading_path=section.full_title, status="running",
                message=f"section {sid}「{section.title}」判为 QA 但抽取为空，回退尝试作为 SOP")
            unit = extract_sop(doc, section, self.llm, cache=self.cache)
            if unit:
                gate_sop(unit)
            if unit and unit.struct_ok:
                sop.append(unit)
                labels[sid] = "sop"      # so coverage checks the SOP set for this sid
                # Drop any empty/invalid QA shells for this section to keep results clean.
                qa = [u for u in qa if not any(p.section_sid == sid for p in u.sources)
                      or u.struct_ok]
                self.bus.publish(
                    "fallback", doc.topic, section=sid, title=section.title,
                    heading_path=section.full_title, status="done",
                    message=f"section {sid}「{section.title}」已回退为 SOP：{unit.title}")
            else:
                reason = unit.struct_reason if unit else "extract_sop 返回空"
                self.bus.publish(
                    "fallback", doc.topic, section=sid, title=section.title,
                    heading_path=section.full_title, status="failed",
                    message=f"section {sid}「{section.title}」回退 SOP 仍失败：{reason}")
        return qa, sop

    @staticmethod
    def _assert_extraction_coverage(secs, labels, qa, sop) -> None:
        """Every non-skip source section must yield a structurally valid unit.

        An empty extraction or a structurally broken retry is a visible document
        failure, not a successful partial result. This is the deterministic
        completeness invariant behind the exported knowledge base.
        """
        qa_sids = {p.section_sid for u in qa if u.struct_ok for p in u.sources}
        sop_sids = {p.section_sid for u in sop if u.struct_ok for p in u.sources}
        gaps = []
        for section in secs:
            label = labels.get(section.sid, "qa")
            if label == "qa" and section.sid not in qa_sids:
                gaps.append(f"{section.sid}:qa")
            elif label == "sop" and section.sid not in sop_sids:
                gaps.append(f"{section.sid}:sop")
        if gaps:
            raise ValueError("extraction coverage gap: " + ", ".join(gaps))

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
        self._failed_this_run.clear()
        usage_before = self.llm.total_usage
        before = (usage_before.prompt_tokens, usage_before.completion_tokens,
                  usage_before.total_tokens, usage_before.calls)
        self.bus.publish("run_status", status="started")
        md_paths = discover_topics(self.settings.input_dir)
        topics = {p: _topic_for(p, self.settings.input_dir) for p in md_paths}
        # Defend against same-named files in different branches without changing
        # familiar keys in the normal case.
        counts: dict[str, int] = {}
        for topic in topics.values():
            counts[topic] = counts.get(topic, 0) + 1
        for path, topic in list(topics.items()):
            if counts[topic] > 1:
                import hashlib
                suffix = hashlib.sha256(str(path.relative_to(self.settings.input_dir)).encode()).hexdigest()[:8]
                topics[path] = f"{topic}__{suffix}"
        if only:
            selected = set(only)
            md_paths = [p for p in md_paths if topics[p] in selected]

        # Seed manifest entries and decide skips.
        todo: list[Path] = []
        for p in md_paths:
            topic = topics[p]
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

        # Sample the LLM scheduler's live concurrency and publish it to the event
        # journal ~1/s so the dashboard can show, cross-process, how many calls are
        # in flight vs. the adaptive window's current limit. Covers the whole run
        # (doc pool AND the review/consolidation phase, both of which hit the gateway).
        sampler_stop = threading.Event()

        def _sample_concurrency():
            while not sampler_stop.is_set():
                try:
                    snap = self.llm.concurrency_stats()
                    self.bus.publish("concurrency", **snap)
                except Exception:      # observability must never break a run
                    pass
                sampler_stop.wait(1.0)

        sampler = threading.Thread(target=_sample_concurrency, daemon=True)
        sampler.start()

        try:
            with ThreadPoolExecutor(max_workers=self.settings.max_workers) as pool:
                futures = {pool.submit(self._process_doc, p, topics[p]): p for p in todo}
                for future in as_completed(futures):
                    res = future.result()
                    # A failed retry must not erase the last known-good snapshot.
                    if not res.error or res.topic not in self._results:
                        with self._lock:
                            self._results[res.topic] = res
            # Persist the per-doc extraction snapshot BEFORE consolidation, so a
            # crash/quota-exhaustion during the (long, network-heavy) review phase
            # never discards work that already succeeded.
            self._persist_results()
            self._consolidate()
            self._persist_results()
            usage = self.llm.total_usage
            self.bus.publish(
                "run_status", status="done", calls=usage.calls - before[3],
                prompt_tokens=usage.prompt_tokens - before[0],
                completion_tokens=usage.completion_tokens - before[1],
                total_tokens=usage.total_tokens - before[2])
            return dict(self._results)
        finally:
            sampler_stop.set()
            # Best-effort final persist: whatever units are in memory (including any
            # already-approved by a partial review before an exception) must reach
            # results.json. Never let an exception on the happy path lose reviewed
            # work — the user's hard rule: "跑完过的成品一定要保留". Guarded so a
            # persist failure can't mask the original error.
            try:
                self._persist_results()
            except Exception:  # noqa: BLE001 - observability/persist must not raise here
                log.exception("final persist failed")
            # Emit a final zeroed snapshot so the dashboard doesn't leave a stale
            # in-flight count showing after the run ends.
            self.bus.publish("concurrency", in_flight=0,
                             **{k: v for k, v in self.llm.concurrency_stats().items()
                                if k != "in_flight"})
            self._running = False

    def _consolidate(self) -> None:
        """Aggregate, semantically review, and dedup all publishable units."""
        all_qa: list[QAUnit] = []
        for res in self._results.values():
            all_qa.extend([u for u in res.qa if u.struct_ok])
        # Cross-file aggregate, review, one source-grounded regeneration, re-review.
        aggregated = aggregate_by_topic(all_qa)
        pending_review = [u for u in aggregated if u.semantic_ok is None]
        self.bus.publish("log", message=f"aggregated {len(all_qa)}→{len(aggregated)} QA; "
                                        f"reviewing {len(pending_review)} changed units")
        def _review_stage(unit, stage, status, detail):
            topic = unit.sources[0].topic if unit.sources else ""
            self.bus.publish("stage", topic, stage=stage, status=status,
                             attempt=unit.review_attempts, detail=detail)

        # Review QA then SOP. Each phase is guarded: an exception (or quota) in one
        # must not discard the other's results, nor the redistribution/persist that
        # commits already-approved units to disk. review_* already degrade under
        # quota (keep + flag), so this catch is the backstop for anything unexpected.
        try:
            review_with_regeneration(
                pending_review, self.llm,
                max_attempts=self.settings.review_regeneration_attempts,
                reviewer_model=self.settings.reviewer_model or None,
                on_stage=_review_stage)
        except Exception:  # noqa: BLE001 - never lose reviewed work to one phase failing
            log.exception("QA review phase failed; keeping partial verdicts")

        all_sop = [u for res in self._results.values() for u in res.sop if u.struct_ok]
        pending_sop_review = [u for u in all_sop if u.semantic_ok is None]
        self.bus.publish(
            "log", message=f"reviewing {len(pending_sop_review)} changed SOP units")
        try:
            review_sop_with_regeneration(
                pending_sop_review, self.llm,
                max_attempts=self.settings.review_regeneration_attempts,
                reviewer_model=self.settings.reviewer_model or None,
                on_stage=_review_stage)
        except Exception:  # noqa: BLE001
            log.exception("SOP review phase failed; keeping partial verdicts")

        survivors = [u for u in aggregated if u.semantic_ok]
        failed_review = [u for u in aggregated if not u.semantic_ok]
        rejected_by_topic: dict[str, int] = {}
        for unit in aggregated:
            if unit.semantic_ok:
                continue
            topic = unit.sources[0].topic if unit.sources else ""
            rejected_by_topic[topic] = rejected_by_topic.get(topic, 0) + 1
        deduped = dedup_qa(survivors)
        self.bus.publish("log",
                         message=f"review kept {len(survivors)}/{len(aggregated)}; dedup→{len(deduped)}")
        # Redistribute consolidated QA back to their primary topic for per-doc stats.
        by_topic: dict[str, list[QAUnit]] = {}
        for u in deduped:
            topic = u.sources[0].topic if u.sources else ""
            by_topic.setdefault(topic, []).append(u)
        for u in failed_review:
            topic = u.sources[0].topic if u.sources else ""
            by_topic.setdefault(topic, []).append(u)
        for topic, res in self._results.items():
            res.qa = by_topic.get(topic, [])
            if topic in self._failed_this_run:
                continue
            self.manifest.update(
                topic, status="done",
                extracted=len(res.qa),
                passed=sum(1 for u in res.qa if u.publication_status == "approved"),
                rejected=rejected_by_topic.get(topic, 0),
                needs_review=sum(1 for u in res.qa if u.needs_review),
                sop_count=sum(1 for u in res.sop
                              if u.struct_ok and u.semantic_ok is not False))
        # Stash the fully consolidated set for export.
        self._consolidated_qa = deduped

    def _persist_results(self) -> None:
        out = self.settings.output_dir / "results.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        payload = {"qa": [], "sop": []}
        seen = set()
        for res in self._results.values():
            for u in res.qa:
                # Dedup key is (module, unit_id), NOT unit_id alone: dedup_qa keeps
                # cross-module twins on purpose (same question, module-specific
                # answer — user's recall-first rule). Keying on unit_id (a hash of
                # the query only) would collapse those twins and silently drop one
                # module's distinct answer here, diverging results.json from the
                # exported set.
                mod = u.sources[0].topic if u.sources else ""
                key = (mod, u.publication_status, u.unit_id)
                if key in seen:
                    continue
                seen.add(key)
                payload["qa"].append({
                    "query": u.query, "answer": u.answer,
                    "paraphrases": u.paraphrases,
                    "needs_review": u.needs_review,
                    "semantic_reason": u.semantic_reason,
                    "semantic_ok": u.semantic_ok,
                    "review_attempts": u.review_attempts,
                    "publication_status": u.publication_status,
                    "review_history": u.review_history,
                    "sources": [s.to_dict(include_excerpt=True) for s in u.sources]})
            for u in res.sop:
                payload["sop"].append({
                    "title": u.title, "markdown": u.markdown,
                    "entry_questions": u.entry_questions,
                    "struct_ok": u.struct_ok,
                    "struct_reason": u.struct_reason,
                    "semantic_ok": u.semantic_ok,
                    "semantic_reason": u.semantic_reason,
                    "needs_review": u.needs_review,
                    "review_attempts": u.review_attempts,
                    "publication_status": u.publication_status,
                    "review_history": u.review_history,
                    "sources": [s.to_dict(include_excerpt=True) for s in u.sources]})
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
        stats = export_all(qa, sop, self.settings.output_dir, include_paraphrases=False)
        self.bus.publish("run_status", status="exported",
                         qa_units=stats.qa_units, qa_rows=stats.qa_rows,
                         sop_files=stats.sop_files, needs_review=stats.needs_review)
        return stats


def _sha_of(path: Path) -> str:
    return source_bundle_sha(path)
