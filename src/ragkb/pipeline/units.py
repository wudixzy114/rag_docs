"""Knowledge-unit data model — the pipeline's OUTPUT types.

Two unit kinds, matching the two ingest formats:
- `QAUnit`: one Query→Answer pair. Exports to the strict 2-column CSV. Carries
  `paraphrases` (extra keys for the same answer — recall booster) and provenance
  (source topic/doc/section) that lives ONLY in the sidecar metadata.jsonl,
  never in the CSV.
- `SOPUnit`: one cleaned procedure/explanation as whole Markdown, plus
  `entry_questions` (user-phrased symptom queries that route to it).

Provenance rationale (user asked): under aggressive cross-file merging, a unit
may draw from several sources. Provenance is how we (a) trace a wrong answer back
to its origin, (b) re-run only affected units when a source doc changes, and (c)
let a human reviewer verify a merge. It is metadata, not query content.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field


@dataclass
class Provenance:
    topic: str = ""               # folder / topic
    doc_title: str = ""
    heading_path: str = ""        # "H1 / H2 / H3"
    section_sid: str = ""         # dotted id within doc
    source_sha: str = ""          # sha of the source file, for incremental re-run
    image_refs: list[str] = field(default_factory=list)
    source_excerpt: str = ""       # masked source evidence used by semantic review

    def to_dict(self, include_excerpt: bool = False) -> dict:
        data = {"topic": self.topic, "doc_title": self.doc_title,
                "heading_path": self.heading_path, "section_sid": self.section_sid,
                "source_sha": self.source_sha, "image_refs": self.image_refs}
        if include_excerpt and self.source_excerpt:
            data["source_excerpt"] = self.source_excerpt
        return data


@dataclass
class QAUnit:
    query: str
    answer: str
    paraphrases: list[str] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    # Quality-gate bookkeeping (not exported to CSV).
    truncated: bool = False
    struct_ok: bool = True        # Layer1 deterministic gate
    struct_reason: str = ""
    semantic_ok: bool | None = None   # Layer2 strong-model gate (None = not yet run)
    semantic_reason: str = ""
    needs_review: bool = False
    review_attempts: int = 0
    publication_status: str = "pending"  # pending|approved|failed_review
    review_history: list[str] = field(default_factory=list)

    def query_keys(self) -> list[str]:
        """All query strings that should point at this answer (main + paraphrases),
        de-duped, order-stable — every one becomes a row in the CSV."""
        seen, out = set(), []
        for q in [self.query, *self.paraphrases]:
            q = (q or "").strip()
            if q and q not in seen:
                seen.add(q)
                out.append(q)
        return out

    @property
    def unit_id(self) -> str:
        return hashlib.sha256(("QA::" + self.query.strip()).encode("utf-8")).hexdigest()[:16]


@dataclass
class SOPUnit:
    title: str
    markdown: str
    entry_questions: list[str] = field(default_factory=list)
    sources: list[Provenance] = field(default_factory=list)
    truncated: bool = False
    struct_ok: bool = True
    struct_reason: str = ""
    semantic_ok: bool | None = None
    semantic_reason: str = ""
    needs_review: bool = False

    @property
    def unit_id(self) -> str:
        return hashlib.sha256(("SOP::" + self.title.strip()).encode("utf-8")).hexdigest()[:16]
