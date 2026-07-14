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
import queue
import threading
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from sse_starlette.sse import EventSourceResponse

from ragkb.config import get_settings
from ragkb.pipeline.events import EventBus
from ragkb.pipeline.orchestrator import Orchestrator

_WEB_DIR = Path(__file__).parent / "web"


def create_app() -> FastAPI:
    app = FastAPI(title="诊断知识库流水线")
    settings = get_settings()
    bus = EventBus()
    orch = Orchestrator(settings=settings, bus=bus)
    run_lock = threading.Lock()
    state = {"running": False}

    def _run(only=None, force=False):
        with run_lock:
            if state["running"]:
                return
            state["running"] = True
        try:
            orch.run(only=only, force=force)
        finally:
            state["running"] = False

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
                results = json.loads(results_path.read_text("utf-8"))
            except (json.JSONDecodeError, ValueError):
                results = {}
        return JSONResponse({"docs": docs, "results": results,
                             "running": state["running"]})

    @app.get("/api/events")
    async def api_events(request: Request):
        q = bus.subscribe()
        # Replay recent history so a late browser catches up.
        backlog = [ev.to_dict() for ev in bus.history()[-200:]]

        async def gen():
            try:
                for ev in backlog:
                    yield {"data": json.dumps(ev, ensure_ascii=False)}
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        ev = q.get(timeout=1.0)
                        yield {"data": json.dumps(ev.to_dict(), ensure_ascii=False)}
                    except queue.Empty:
                        yield {"event": "ping", "data": "{}"}
            finally:
                bus.unsubscribe(q)

        return EventSourceResponse(gen())

    @app.post("/api/run")
    async def api_run(request: Request):
        body = await _json(request)
        only = body.get("only")
        force = bool(body.get("force", False))
        threading.Thread(target=_run, kwargs={"only": only, "force": force},
                         daemon=True).start()
        return {"ok": True}

    @app.post("/api/export")
    def api_export():
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
        threading.Thread(target=_run, kwargs={"only": topics, "force": True},
                         daemon=True).start()
        return {"ok": True}

    return app


async def _json(request: Request) -> dict:
    try:
        return await request.json()
    except (json.JSONDecodeError, ValueError):
        return {}


app = create_app()
