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
from ragkb.parse.source import load_source
from ragkb.pipeline.aggregate import aggregate_by_topic
from ragkb.pipeline.audit import AuditStore
from ragkb.pipeline.classify import classify_sections
from ragkb.pipeline.dedup import dedup_qa
from ragkb.pipeline.events import EventBus, read_journal
from ragkb.pipeline.control import (
    DecisionStore, HumanReviewRequired, inspect_preflight,
)
from ragkb.pipeline.failures import classify_failure, sanitize_failure_message
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
from ragkb.store.manifest import PIPELINE_VERSION, DocState, Manifest
from ragkb.pipeline.scrub import save_mapping

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
    if path.stem.lower() in {"原始文档", "index"}:
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
        self.decisions = DecisionStore(self.settings.output_dir / "decisions.json")
        self.audit = AuditStore(self.settings.output_dir / "audit")
        self._results: dict[str, DocResult] = {}
        self._lock = threading.Lock()
        self._running = False
        self._failed_this_run: set[str] = set()
        recovered = self.manifest.recover_interrupted()
        for topic in recovered:
            self.bus.publish("doc_status", topic, status="interrupted")
        if recovered:
            self.bus.publish("run_status", status="interrupted",
                             message=f"recovered {len(recovered)} stale running documents")
        self._migrate_failure_states()
        self._load_previous_results()
        self._bootstrap_review_decisions()

    def _migrate_failure_states(self) -> None:
        """Classify legacy failures so every old item has an actionable policy."""
        changes: dict[str, dict] = {}
        for item in self.decisions.all():
            safe_reason = sanitize_failure_message(item.reason)
            if safe_reason != item.reason:
                self.decisions.propose(
                    item.topic, item.stage, reason=safe_reason,
                    evidence=item.evidence,
                    recommended_action=item.recommended_action,
                    options=item.options)
        latest_errors: dict[str, dict] = {}
        events, _ = read_journal(self.settings.output_dir / "events.jsonl", 0)
        for event in events:
            if event.get("kind") == "error" and event.get("topic"):
                latest_errors[event["topic"]] = event.get("data") or {}
        for state in self.manifest.all():
            if state.status == "interrupted":
                changes[state.topic] = {
                    "error_code": "interrupted", "retryable": True,
                    "error": state.error or "上次进程中断，可从缓存安全重试"}
                continue
            if (state.status == "awaiting_review" and state.error_code == "content_blocked"
                    and state.current_stage in {"", "unknown"}):
                stage = self._infer_failure_stage(state.error)
                if state.decision_id:
                    self.decisions.remove(state.decision_id)
                decision = self.decisions.propose(
                    state.topic, stage, reason=state.error,
                    evidence=["content_blocked"], recommended_action="retry",
                    options=["retry", "exclude"])
                changes[state.topic] = {"current_stage": stage,
                                        "decision_id": decision.decision_id}
                continue
            if state.status == "awaiting_review" and state.error_code == "content_blocked":
                safe_error = sanitize_failure_message(state.error)
                decision = self.decisions.propose(
                    state.topic, state.current_stage or "extract", reason=safe_error,
                    evidence=["content_blocked"], recommended_action="retry",
                    options=["retry", "exclude"])
                changes[state.topic] = {"error": safe_error,
                                        "decision_id": decision.decision_id}
                continue
            if state.status != "failed":
                continue
            journal_error = latest_errors.get(state.topic, {})
            error = state.error or str(journal_error.get("message") or "")
            if state.error_code not in {"", "pipeline_error"} and state.error:
                continue
            code, retryable = classify_failure(RuntimeError(error))
            current_stage = "" if state.current_stage == "unknown" else state.current_stage
            stage = (current_stage or str(journal_error.get("stage") or "")
                     or self._infer_failure_stage(error))
            changes[state.topic] = {
                "error_code": code, "retryable": retryable,
                "current_stage": stage, "error": error or "历史任务失败，可重试"}
            if not retryable:
                decision = self.decisions.propose(
                    state.topic, stage, reason=error, evidence=[code],
                    recommended_action="retry", options=["retry", "exclude"])
                changes[state.topic].update({
                    "status": "awaiting_review", "decision_id": decision.decision_id})
        self.manifest.bulk_update(changes)

    @staticmethod
    def _infer_failure_stage(error: str) -> str:
        text = (error or "").lower()
        if "vision" in text or "image" in text:
            return "vision"
        if "classif" in text:
            return "classify"
        if "extract" in text or "json" in text:
            return "extract"
        if "coverage" in text or "truncat" in text:
            return "validate"
        if "review" in text or "regenerat" in text:
            return "review"
        return "unknown"

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

    def reload_results(self) -> None:
        """Refresh the durable result snapshot produced by another process."""
        with self._lock:
            self._results = {}
            self._consolidated_qa = []
            self._load_previous_results()

    def _bootstrap_review_decisions(self) -> None:
        """Expose failed-review units from an older results.json to operators."""
        changes: dict[str, dict] = {}
        for topic, result in self._results.items():
            state = self.manifest.get(topic)
            if state and state.status == "excluded":
                continue
            failed = [unit for unit in [*result.qa, *result.sop] if unit.needs_review]
            if not failed:
                continue
            transient = all(self._transient_review_failure(unit) for unit in failed)
            if transient:
                old = self.decisions.get_for(topic, "review")
                if old and old.status == "pending":
                    self.decisions.remove(old.decision_id)
                fields = {"needs_review": len(failed)}
                if state and state.status in {"done", "partial"}:
                    fields.update({
                        "retryable": True, "error_code": "review_incomplete",
                        "error": "自动审核因模型/配额不可用而未完成，将在下一轮自动重试"})
                elif (state and state.status == "awaiting_review"
                      and state.preflight_status == "awaiting_review"):
                    fields.update({
                        "retryable": True, "error_code": "preflight_review",
                        "error": "等待预处理人工决策"})
                if state and state.decision_id == (old.decision_id if old else ""):
                    fields["decision_id"] = ""
                if state and state.status == "done":
                    fields["status"] = "partial"
                changes[topic] = fields
                decision_id = ""
            else:
                evidence = [
                    f"{unit.unit_id}:{unit.semantic_reason}" for unit in failed[:100]]
                decision = self.decisions.propose(
                    topic, "review",
                    reason=f"{len(failed)} 个历史单元未通过自动语义审核",
                    evidence=evidence, recommended_action="retry",
                    options=["retry", "accept", "exclude"])
                decision_id = decision.decision_id
                if decision.status == "pending":
                    changes[topic] = {"decision_id": decision.decision_id,
                                      "needs_review": len(failed)}
            audit = self.audit.get(topic)
            if not audit.get("stages", {}).get("recovered_results"):
                source = "\n\n".join(
                    provenance.source_excerpt
                    for unit in failed for provenance in unit.sources
                    if provenance.source_excerpt)
                self.audit.record(
                    topic, "recovered_results", status="needs_review",
                    input_text=source, output_text=self._units_text(result.qa, result.sop),
                    metadata={"needs_review": len(failed),
                              "decision_id": decision_id,
                              "note": "历史运行没有逐阶段快照；由来源证据和结果快照恢复"})
        self.manifest.bulk_update(changes)

    @staticmethod
    def _transient_review_failure(unit) -> bool:
        reason = unit.semantic_reason or ""
        return reason.startswith(("regeneration_failed:no_result",
                                  "review_unavailable:", "no_verdict"))

    # ---- per-document pipeline -------------------------------------------
    def _process_doc(self, md_path: Path, topic: str | None = None) -> DocResult:
        topic = topic or _topic_for(md_path, self.settings.input_dir)
        self.bus.publish("doc_status", topic, status="running")
        previous = self.manifest.get(topic)
        attempts = (previous.attempts if previous else 0) + 1
        self.manifest.update(topic, status="running", error="", error_code="",
                             retryable=False, attempts=attempts,
                             current_stage="parse")
        current_stage = "parse"
        try:
            self.bus.publish("stage", topic, stage="parse", status="running")
            source_text = load_source(md_path)
            doc = parse_document(md_path, topic=topic)
            self.manifest.update(topic, source_sha=doc.source_sha)
            parsed_text = self._sections_text(list(doc.iter_sections()))
            self.audit.record(topic, "parse", status="done", input_text=source_text,
                              output_text=parsed_text,
                              metadata={"source_path": str(md_path),
                                        "sections": sum(1 for _ in doc.iter_sections())})
            self.manifest.update(topic, last_completed_stage="parse",
                                 current_stage="vision")
            self.bus.publish("stage", topic, stage="parse", status="done",
                             detail=f"{sum(1 for _ in doc.iter_sections())} sections")
            # 1. Vision-read every image (replaces weak inline OCR).
            current_stage = "vision"
            imgs = doc.all_images()
            vision_before = "\n\n".join(
                f"[{im.rel_path}]\n{im.inline_ocr}" for im in imgs)
            self.bus.publish("stage", topic, stage="vision", status="running",
                             done=0, total=len(imgs))
            for k, im in enumerate(imgs):
                vision_read_image(im, self.llm, self.cache)
                self.bus.publish("doc_progress", topic, stage="vision",
                                 done=k + 1, total=len(imgs))
                self.bus.publish("stage", topic, stage="vision", status="running",
                                 done=k + 1, total=len(imgs))
            failed_images = [im.rel_path for im in imgs if im.vision_failed]
            vision_after = "\n\n".join(
                f"[{im.rel_path}]\n{im.vision_text or im.inline_ocr}" for im in imgs)
            self.audit.record(
                topic, "vision", status="needs_review" if failed_images else "done",
                input_text=vision_before, output_text=vision_after,
                metadata={"images": len(imgs), "failed_images": failed_images})
            if failed_images:
                decision = self.decisions.propose(
                    topic, "vision", reason="图片模型重试后仍有未读取内容",
                    evidence=failed_images, recommended_action="retry",
                    options=["retry", "accept", "exclude"])
                if decision.status == "pending":
                    raise HumanReviewRequired(
                        f"{len(failed_images)} 张图片读取失败，需要选择重试或接受 OCR",
                        decision.decision_id)
                if decision.selected_action == "exclude":
                    raise HumanReviewRequired("人工决定排除该文档", decision.decision_id)
                if decision.selected_action == "retry":
                    raise RuntimeError("vision failed after retries: " + ", ".join(failed_images))
            self.bus.publish("stage", topic, stage="vision", status="done",
                             done=len(imgs), total=len(imgs))
            self.manifest.update(topic, last_completed_stage="vision",
                                 current_stage="classify")
            # 2. Classify sections.
            current_stage = "classify"
            secs = split_oversize_sections(
                [s for s in doc.iter_sections() if s.body.strip() or s.images])
            self.bus.publish("stage", topic, stage="classify", status="running",
                             done=0, total=len(secs))
            labels = classify_sections(secs, self.llm)
            self.audit.record(
                topic, "classify", status="done",
                input_text=self._sections_text(secs),
                output_text=json.dumps(labels, ensure_ascii=False, indent=2),
                metadata={"labels": {label: list(labels.values()).count(label)
                                     for label in set(labels.values())}})
            self.bus.publish("stage", topic, stage="classify", status="done",
                             done=len(secs), total=len(secs))
            self.manifest.update(topic, last_completed_stage="classify",
                                 current_stage="extract")
            # 3. Extract with size-aware batches. Documents are the concurrency
            #    boundary, so there is no nested pool multiplying gateway pressure.
            current_stage = "extract"
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
            extracted_text = self._units_text(qa, sop)
            self.audit.record(
                topic, "extract", status="done",
                input_text=self._sections_text(secs), output_text=extracted_text,
                metadata={"qa": len(qa), "sop": len(sop)})
            self.manifest.update(topic, last_completed_stage="extract",
                                 current_stage="validate")
            # 4. Layer 1 structural gate (retry truncated once with more tokens).
            current_stage = "validate"
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
            self.audit.record(
                topic, "validate", status="done", input_text=extracted_text,
                output_text=self._units_text(qa, sop),
                metadata={"qa_valid": sum(u.struct_ok for u in qa),
                          "sop_valid": sum(u.struct_ok for u in sop)})
            self.bus.publish("stage", topic, stage="validate", status="done",
                             detail=(f"{sum(u.struct_ok for u in qa)} QA / "
                                     f"{sum(u.struct_ok for u in sop)} SOP valid"))
            # Optional experiment only; the production path remains source-faithful.
            ok_units = [u for u in qa if u.struct_ok]
            if self.settings.enable_paraphrases and ok_units:
                add_paraphrases_batch(ok_units, self.llm, self.cache)
            res = DocResult(topic=topic, qa=qa, sop=sop)
            self._publish_units(topic, qa, sop)
            self.manifest.update(topic, last_completed_stage="validate",
                                 current_stage="review")
            return res
        except HumanReviewRequired as exc:
            decision = self.decisions.get_for(topic, current_stage)
            excluded = decision and decision.selected_action == "exclude"
            status = "excluded" if excluded else "awaiting_review"
            self.audit.record(topic, current_stage, status=status,
                              error=str(exc),
                              metadata={"decision_id": exc.decision_id})
            self.manifest.update(
                topic, status=status, error=str(exc), error_code="human_review_required",
                retryable=not excluded, current_stage=current_stage,
                decision_id=exc.decision_id)
            self.bus.publish("decision_required", topic, stage=current_stage,
                             decision_id=exc.decision_id, message=str(exc))
            with self._lock:
                self._failed_this_run.add(topic)
            return DocResult(topic=topic, error=str(exc))
        except Exception as exc:                      # noqa: BLE001 - isolate one doc's failure
            log.exception("doc %s failed", topic)
            error_code, retryable = classify_failure(exc)
            safe_error = sanitize_failure_message(str(exc))
            raw_output = getattr(exc, "raw_text", "")
            self.audit.record(topic, current_stage, status="failed",
                              output_text=raw_output, error=safe_error,
                              metadata={"error_code": error_code,
                                        "retryable": retryable})
            decision_id = ""
            if not retryable:
                decision = self.decisions.propose(
                    topic, current_stage, reason=safe_error, evidence=[error_code],
                    recommended_action="retry", options=["retry", "exclude"])
                decision_id = decision.decision_id
            status = "awaiting_review" if decision_id else "failed"
            self.bus.publish("error", topic, message=safe_error, code=error_code,
                             retryable=retryable, stage=current_stage)
            self.manifest.update(topic, status=status, error=safe_error,
                                 error_code=error_code, retryable=retryable,
                                 current_stage=current_stage, decision_id=decision_id)
            self.bus.publish("stage", topic, stage=current_stage, status="failed",
                             detail=safe_error, error_code=error_code)
            with self._lock:
                self._failed_this_run.add(topic)
            return DocResult(topic=topic, error=str(exc))
        finally:
            # results.json contains placeholders, so the matching reversible map
            # must survive a crash or an export from another process.
            try:
                save_mapping(self.settings.output_dir / "redaction_map.json")
            except OSError:
                log.exception("failed to persist redaction mapping")

    @staticmethod
    def _sections_text(sections) -> str:
        return "\n\n".join(
            f"## [{section.sid}] {section.full_title}\n{section.body}".strip()
            for section in sections)

    @staticmethod
    def _units_text(qa, sop) -> str:
        payload = {
            "qa": [{"query": unit.query, "answer": unit.answer,
                    "struct_ok": unit.struct_ok,
                    "struct_reason": unit.struct_reason} for unit in qa],
            "sop": [{"title": unit.title, "markdown": unit.markdown,
                     "entry_questions": unit.entry_questions,
                     "struct_ok": unit.struct_ok,
                     "struct_reason": unit.struct_reason} for unit in sop],
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

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

    def _discovered_paths(self) -> tuple[list[Path], dict[Path, str]]:
        paths = discover_topics(self.settings.input_dir)
        topics = {path: _topic_for(path, self.settings.input_dir) for path in paths}
        counts: dict[str, int] = {}
        for topic in topics.values():
            counts[topic] = counts.get(topic, 0) + 1
        for path, topic in list(topics.items()):
            if counts[topic] > 1:
                import hashlib
                relative = str(path.relative_to(self.settings.input_dir))
                suffix = hashlib.sha256(relative.encode()).hexdigest()[:8]
                topics[path] = f"{topic}__{suffix}"
        return paths, topics

    def scan_preflight(self, only: list[str] | None = None, *, force: bool = False) -> dict:
        """Run the no-LLM document quarantine pass and persist every verdict."""
        paths, topics = self._discovered_paths()
        selected = set(only or [])
        counts = {"included": 0, "awaiting_review": 0, "excluded": 0}
        for path in paths:
            topic = topics[path]
            if selected and topic not in selected:
                continue
            sha = _sha_of(path)
            state = self.manifest.get(topic)
            if state is None:
                state = DocState(topic=topic, source_sha=sha)
                self.manifest.upsert(state)
            if (not force and state.preflight_status != "pending"
                    and state.source_sha == sha):
                counts[state.preflight_status] = counts.get(state.preflight_status, 0) + 1
                continue
            flags, evidence = inspect_preflight(path)
            status = "included"
            decision_id = ""
            if flags:
                decision = self.decisions.propose(
                    topic, "preflight",
                    reason="文档具有明确的私密/废弃标记，进入流水线前需确认",
                    evidence=evidence, recommended_action="exclude",
                    options=["include", "exclude"])
                decision_id = decision.decision_id
                if decision.status == "pending":
                    status = "awaiting_review"
                else:
                    status = "excluded" if decision.selected_action == "exclude" else "included"
            manifest_status = state.status
            if status in {"awaiting_review", "excluded"}:
                manifest_status = status
            elif manifest_status in {"awaiting_review", "excluded"}:
                manifest_status = "pending"
            update = {
                "source_sha": sha, "status": manifest_status,
                "preflight_status": status, "preflight_flags": flags,
            }
            if status == "awaiting_review":
                update.update({
                    "decision_id": decision_id, "error": "等待预处理人工决策",
                    "error_code": "preflight_review", "retryable": True})
            elif status == "excluded":
                update.update({
                    "decision_id": decision_id, "error": "预处理决定排除",
                    "error_code": "", "retryable": False})
            elif state.error_code == "preflight_review":
                update.update({"decision_id": "", "error": "", "error_code": "",
                               "retryable": False})
            self.manifest.update(topic, **update)
            try:
                source = load_source(path)
            except (OSError, ValueError):
                source = ""
            self.audit.record(
                topic, "preflight", status=status, input_text=source,
                output_text=json.dumps({"flags": flags, "evidence": evidence,
                                        "status": status}, ensure_ascii=False, indent=2),
                metadata={"source_path": str(path), "decision_id": decision_id})
            self.bus.publish("stage", topic, stage="preflight", status=status,
                             flags=flags, decision_id=decision_id)
            if status == "awaiting_review":
                self.bus.publish("decision_required", topic, stage="preflight",
                                 decision_id=decision_id,
                                 message="请确认包含私密/废弃标记的文档是否进入流水线")
            counts[status] = counts.get(status, 0) + 1
        return counts

    def resolve_decision(self, decision_id: str, action: str) -> dict:
        """Apply an operator decision to both control state and publishable data."""
        decision = self.decisions.resolve(decision_id, action)
        topic = decision.topic
        if decision.stage == "review" or action == "exclude":
            self.reload_results()
        if action == "exclude":
            result = self._results.get(topic)
            if result:
                for unit in [*result.qa, *result.sop]:
                    unit.semantic_ok = False
                    unit.needs_review = True
                    unit.publication_status = "failed_review"
                    unit.semantic_reason = "human_excluded"
                self._persist_results()
            self.decisions.close_topic(topic, "exclude", except_id=decision_id)
            self.manifest.update(
                topic, status="excluded", decision_id="", retryable=False,
                preflight_status=("excluded" if decision.stage == "preflight"
                                  else (self.manifest.get(topic).preflight_status
                                        if self.manifest.get(topic) else "included")),
                error="人工决定排除", error_code="")
        elif decision.stage == "preflight":
            self.manifest.update(
                topic, status="pending", preflight_status="included",
                decision_id="", error="", error_code="", retryable=False)
        elif decision.stage == "review" and action == "accept":
            result = self._results.get(topic)
            accepted = 0
            if result:
                for unit in [*result.qa, *result.sop]:
                    if not unit.needs_review:
                        continue
                    unit.semantic_ok = True
                    unit.needs_review = False
                    unit.publication_status = "approved"
                    unit.review_history.append("human_accept")
                    unit.semantic_reason = "human_accept"
                    accepted += 1
                self._persist_results()
            self.manifest.update(topic, status="done", decision_id="",
                                 retryable=False, needs_review=0,
                                 passed=sum(1 for unit in result.qa
                                            if unit.publication_status == "approved") if result else 0,
                                 sop_count=sum(1 for unit in result.sop
                                               if unit.publication_status == "approved") if result else 0,
                                 error="", error_code="")
            self.bus.publish("human_decision", topic, stage=decision.stage,
                             action=action, accepted=accepted)
            return {"decision": decision.to_dict(), "accepted": accepted}
        else:
            self.manifest.update(topic, status="pending", decision_id="",
                                 retryable=True, error="", error_code="")
        self.bus.publish("human_decision", topic, stage=decision.stage, action=action)
        return {"decision": decision.to_dict()}

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
        md_paths, topics = self._discovered_paths()
        if only:
            selected = set(only)
            md_paths = [p for p in md_paths if topics[p] in selected]

        preflight = self.scan_preflight(only=only)
        self.bus.publish("preflight_summary", **preflight)

        # Seed manifest entries and decide skips.
        todo: list[Path] = []
        for p in md_paths:
            topic = topics[p]
            sha = _sha_of(p)
            if not self.manifest.get(topic):
                self.manifest.upsert(DocState(topic=topic, source_sha=sha))
            st = self.manifest.get(topic)
            if st and st.preflight_status in {"awaiting_review", "excluded"}:
                self.bus.publish("doc_status", topic, status=st.preflight_status)
                continue
            if st and st.decision_id:
                decision = self.decisions.get(st.decision_id)
                if decision and decision.status == "pending":
                    self.bus.publish("doc_status", topic, status="awaiting_review")
                    continue
                if decision and decision.selected_action == "exclude":
                    self.manifest.update(topic, status="excluded", retryable=False)
                    self.bus.publish("doc_status", topic, status="excluded")
                    continue
                self.manifest.update(topic, decision_id="", status="pending",
                                     error="", error_code="")
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
        except Exception as exc:
            self.bus.publish("run_status", status="failed", message=str(exc))
            raise
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
        extracted_qa_count: dict[str, int] = {}
        for res in self._results.values():
            valid = [u for u in res.qa if u.struct_ok]
            all_qa.extend(valid)
            extracted_qa_count[res.topic] = len(valid)
        # Cross-file aggregate, review, one source-grounded regeneration, re-review.
        aggregated = aggregate_by_topic(all_qa)
        all_sop = [u for res in self._results.values() for u in res.sop if u.struct_ok]
        pre_review_qa: dict[str, list[QAUnit]] = {}
        pre_review_sop: dict[str, list[SOPUnit]] = {}
        for unit in aggregated:
            topic = unit.sources[0].topic if unit.sources else ""
            pre_review_qa.setdefault(topic, []).append(unit)
        for unit in all_sop:
            topic = unit.sources[0].topic if unit.sources else ""
            pre_review_sop.setdefault(topic, []).append(unit)
        pre_review_text = {
            topic: self._units_text(pre_review_qa.get(topic, []),
                                    pre_review_sop.get(topic, []))
            for topic in set(pre_review_qa) | set(pre_review_sop)}
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
            extract_attempts = self.audit.get(topic).get("stages", {}).get("extract", [])
            extract_text = (extract_attempts[-1].get("output_text", "")
                            if extract_attempts else pre_review_text.get(topic, ""))
            review_input = pre_review_text.get(topic, extract_text)
            final_text = self._units_text(res.qa, res.sop)
            self.audit.record(
                topic, "consolidate", status="done",
                input_text=extract_text, output_text=review_input,
                metadata={"qa_before": extracted_qa_count.get(topic, 0),
                          "qa_after": len(pre_review_qa.get(topic, []))})
            regenerated = [unit for unit in [*res.qa, *res.sop]
                           if unit.review_attempts > 0]
            if regenerated:
                self.audit.record(
                    topic, "regenerate", status="done",
                    input_text=review_input, output_text=final_text,
                    metadata={"units": len(regenerated),
                              "max_attempts": max(unit.review_attempts
                                                  for unit in regenerated)})
            review_failures = [
                unit for unit in [*res.qa, *res.sop] if unit.needs_review]
            decision_id = ""
            transient_review = bool(review_failures) and all(
                self._transient_review_failure(unit) for unit in review_failures)
            if review_failures and not transient_review:
                evidence = [
                    f"{unit.unit_id}:{unit.semantic_reason}" for unit in review_failures[:100]]
                decision = self.decisions.propose(
                    topic, "review",
                    reason=f"{len(review_failures)} 个单元未通过自动语义审核",
                    evidence=evidence, recommended_action="retry",
                    options=["retry", "accept", "exclude"])
                decision_id = decision.decision_id if decision.status == "pending" else ""
                if decision.status == "pending":
                    self.bus.publish(
                        "decision_required", topic, stage="review",
                        decision_id=decision.decision_id,
                        message=f"{len(review_failures)} 个单元需要人工复核")
            self.audit.record(
                topic, "review", status="needs_review" if review_failures else "done",
                input_text=review_input, output_text=final_text,
                metadata={"needs_review": len(review_failures),
                          "decision_id": decision_id})
            self.manifest.update(
                topic, status="partial" if transient_review else "done",
                extracted=len(res.qa),
                passed=sum(1 for u in res.qa if u.publication_status == "approved"),
                rejected=rejected_by_topic.get(topic, 0),
                needs_review=sum(1 for u in res.qa if u.needs_review),
                sop_count=sum(1 for u in res.sop
                              if u.struct_ok and u.semantic_ok is not False),
                pipeline_version=PIPELINE_VERSION, current_stage="",
                last_completed_stage="review", retryable=transient_review,
                error=("自动审核未完成，将在下一轮自动重试"
                       if transient_review else ""),
                error_code="review_incomplete" if transient_review else "",
                decision_id=decision_id)
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
