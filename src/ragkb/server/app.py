"""FastAPI dashboard — live observability + control.

Endpoints:
- GET  /                → the single-file dashboard UI
- GET  /api/state       → current manifest + last results snapshot (for cold load)
- GET  /api/events      → SSE stream of pipeline Events (live progress + units)
- POST /api/run         → start a run (optional {only:[...], force:bool}) in a bg thread
- POST /api/export      → write CSV/SOP/metadata from the last run
- POST /api/pin         → {topic, pinned} pin/unpin a doc (pinned never overwritten)
- POST /api/retry       → {topics:[...]} force re-run specific docs

The orchestrator + event bus live for the process lifetime. A run executes in a
background thread so the SSE stream and control endpoints stay responsive.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from ragkb.config import get_settings
from ragkb.pipeline.events import EventBus, read_journal
from ragkb.pipeline.orchestrator import Orchestrator

_WEB_DIR = Path(__file__).parent / "web"
log = logging.getLogger(__name__)


def create_app() -> FastAPI:
    app = FastAPI(title="诊断知识库流水线")
    settings = get_settings()
    # Journal-backed bus: the dashboard reads live state from the on-disk journal,
    # so it reflects runs started HERE (via /api/run) AND runs started from the
    # CLI in a separate process. The journal is the cross-process source of truth.
    journal_path = settings.output_dir / "events.jsonl"
    bus = EventBus(journal_path=journal_path)
    orch = Orchestrator(settings=settings, bus=bus)
    run_lock = threading.Lock()
    state = {"running": False}

    def _run_reserved(only=None, force=False):
        """Execute a run after the API handler atomically reserved the slot."""
        try:
            orch.run(only=only, force=force)
        except Exception as exc:  # surface background failures through SSE/state
            log.exception("pipeline run failed")
            bus.publish("run_status", status="failed", message=str(exc))
        finally:
            with run_lock:
                state["running"] = False

    def _start_run(only=None, force=False) -> bool:
        with run_lock:
            if state["running"]:
                return False
            state["running"] = True
        threading.Thread(target=_run_reserved,
                         kwargs={"only": only, "force": force}, daemon=True).start()
        return True

    @app.get("/", response_class=HTMLResponse)
    def index():
        return (_WEB_DIR / "index.html").read_text("utf-8")

    @app.get("/api/state")
    def api_state():
        docs = [d.to_dict() for d in orch.manifest.all()]
        results_path = settings.output_dir / "results.json"
        results = {}
        if results_path.is_file():
            try:
                results = _public_results(json.loads(results_path.read_text("utf-8")))
            except (json.JSONDecodeError, ValueError):
                results = {}
        # Read events from the on-disk journal (cross-process): reflects runs from
        # the CLI as well as this server. Fall back to in-memory history if the
        # journal isn't present yet.
        journal_events, _ = read_journal(journal_path, 0)
        if journal_events:
            recent = journal_events[-2000:]
        else:
            recent = [event.to_dict() for event in bus.history()]
        stages: dict[str, dict[str, dict]] = {}
        errors = []
        last_run = {}
        concurrency = {}
        for event in recent:
            kind = event.get("kind")
            topic = event.get("topic", "")
            data = event.get("data", {}) or {}
            if kind == "stage" and topic:
                stages.setdefault(topic, {})[data.get("stage", "unknown")] = {
                    **data, "seq": event.get("seq"), "ts": event.get("ts")}
            elif kind == "error":
                errors.append(event)
            elif kind == "run_status":
                last_run = {**data, "seq": event.get("seq"), "ts": event.get("ts")}
            elif kind == "concurrency":
                concurrency = {**data, "ts": event.get("ts")}
        llm_settings = orch.llm.settings
        usage = orch.llm.total_usage
        # `running` reflects a run in ANY process: this server's own flag, or a
        # journal whose latest run_status is 'started' (e.g. a CLI run).
        running = state["running"] or last_run.get("status") == "started"
        # Keep the high-frequency concurrency heartbeat out of the visible event
        # feed; it's surfaced as a live gauge via `concurrency` instead.
        feed = [e for e in recent if e.get("kind") != "concurrency"]
        return JSONResponse({
            "docs": docs, "results": results, "running": running,
            "stages": stages, "errors": errors[-50:], "last_run": last_run,
            "concurrency": concurrency,
            "events": feed[-300:],
            "usage": usage.__dict__,
            "models": {
                "classify": llm_settings.model_for("classify"),
                "extract": llm_settings.model_for("extract"),
                "review": llm_settings.model_for("review"),
            },
        })

    @app.get("/api/events")
    async def api_events(request: Request):
        # Tail the on-disk journal by byte offset rather than only this process's
        # in-memory queue: that's what lets the dashboard stream events produced
        # by ANOTHER process (a CLI `ragkb run`). Start near the end so a fresh
        # connection replays a bounded backlog, then follows appends live.
        import asyncio

        all_events, offset = read_journal(journal_path, 0)
        backlog = all_events[-300:]

        async def gen():
            nonlocal offset
            try:
                for ev in backlog:
                    yield {"data": json.dumps(ev, ensure_ascii=False)}
                idle = 0
                while True:
                    if await request.is_disconnected():
                        break
                    new_events, offset = read_journal(journal_path, offset)
                    if new_events:
                        idle = 0
                        for ev in new_events:
                            yield {"data": json.dumps(ev, ensure_ascii=False)}
                    else:
                        idle += 1
                        if idle >= 5:      # ~a ping every 5 idle polls (5s)
                            idle = 0
                            yield {"event": "ping", "data": "{}"}
                    await asyncio.sleep(1.0)
            except asyncio.CancelledError:
                raise

        return EventSourceResponse(gen())

    @app.post("/api/run")
    async def api_run(request: Request):
        body = await _json(request)
        only = body.get("only")
        force = bool(body.get("force", False))
        if not _start_run(only=only, force=force):
            raise HTTPException(status_code=409, detail="pipeline is already running")
        return {"ok": True}

    @app.post("/api/export")
    def api_export():
        with run_lock:
            if state["running"]:
                raise HTTPException(status_code=409,
                                    detail="cannot export while pipeline is running")
            stats = orch.export()
        return {"ok": True, "stats": stats.__dict__}

    @app.post("/api/pin")
    async def api_pin(request: Request):
        body = await _json(request)
        topic = body.get("topic", "")
        pinned = bool(body.get("pinned", True))
        orch.manifest.set_pinned(topic, pinned)
        bus.publish("doc_status", topic, status="pinned" if pinned else "unpinned")
        return {"ok": True}

    @app.post("/api/retry")
    async def api_retry(request: Request):
        body = await _json(request)
        topics = body.get("topics") or []
        if not _start_run(only=topics, force=True):
            raise HTTPException(status_code=409, detail="pipeline is already running")
        return {"ok": True}

    return app


async def _json(request: Request) -> dict:
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}


def _public_results(data: dict) -> dict:
    """Remove large source evidence while retaining traceable source metadata."""
    clean = {"qa": [], "sop": []}
    for kind in clean:
        for item in data.get(kind, []) or []:
            row = dict(item)
            row["sources"] = [
                {key: value for key, value in source.items() if key != "source_excerpt"}
                for source in item.get("sources", [])]
            clean[kind].append(row)
    return clean


app = create_app()
