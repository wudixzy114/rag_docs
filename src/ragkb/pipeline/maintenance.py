"""Non-destructive maintenance operations for durable pipeline artifacts."""
from __future__ import annotations

import json
import shutil
import time
from pathlib import Path


def _atomic_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), "utf-8")
    tmp.replace(path)


def clean_results(output_dir: Path) -> dict:
    """Quarantine unpublished/invalid rows and leave only approved rows active."""
    output_dir = Path(output_dir)
    source = output_dir / "results.json"
    if not source.is_file():
        raise FileNotFoundError(source)
    original = json.loads(source.read_text("utf-8"))
    stamp = time.strftime("%Y%m%d-%H%M%S")
    snapshot = output_dir / "recovery-snapshots" / stamp
    snapshot.mkdir(parents=True, exist_ok=False)
    copied = ["results.json", "manifest.json", "decisions.json",
              "redaction_map.json", "qa_pairs.csv", "metadata.jsonl",
              "知识库上传包.zip"]
    for name in copied:
        src = output_dir / name
        if src.is_file():
            shutil.copy2(src, snapshot / name)
    for dirname in ("sop", "by_module"):
        src = output_dir / dirname
        if src.is_dir():
            shutil.copytree(src, snapshot / dirname)

    active = {"qa": [], "sop": []}
    quarantine = {"schema_version": 1, "created_at": time.time(),
                  "source": str(source), "qa": [], "sop": [], "invalid": []}
    for kind in ("qa", "sop"):
        for index, row in enumerate(original.get(kind, []) or []):
            if not isinstance(row, dict):
                quarantine["invalid"].append({"kind": kind, "index": index,
                                              "row": row, "reason": "not_object"})
                continue
            valid = (row.get("semantic_ok") is True
                     and row.get("publication_status") == "approved")
            if kind == "sop":
                valid = valid and row.get("struct_ok", True) is True
            if valid:
                active[kind].append(row)
            else:
                quarantine[kind].append(row)
    _atomic_json(snapshot / "quarantine.json", quarantine)
    _atomic_json(output_dir / "quarantine.json", quarantine)
    _atomic_json(source, active)
    return {
        "snapshot": str(snapshot),
        "active_qa": len(active["qa"]), "active_sop": len(active["sop"]),
        "quarantined_qa": len(quarantine["qa"]),
        "quarantined_sop": len(quarantine["sop"]),
        "invalid": len(quarantine["invalid"]),
    }
