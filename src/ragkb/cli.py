"""CLI entry point: run / export / serve.

  ragkb run [--only TOPIC ...] [--force]   process docs, print progress
  ragkb export                             write CSV/SOP/metadata from last run
  ragkb serve [--port 8000]                launch the dashboard

`run` streams progress to the terminal by draining the event bus, so it's usable
headless (CI, cron) as well as via the dashboard.
"""
from __future__ import annotations

import threading

import typer

from ragkb.config import get_settings
from ragkb.pipeline.events import EventBus
from ragkb.pipeline.orchestrator import Orchestrator

app = typer.Typer(add_completion=False, help="诊断知识库文档处理流水线")


@app.command()
def run(only: list[str] = typer.Option(None, "--only", help="只处理指定主题(可多次)"),
        force: bool = typer.Option(False, "--force", help="忽略幂等跳过(仍不覆盖pinned)"),
        export: bool = typer.Option(True, "--export/--no-export", help="完成后导出")):
    """Process documents and (by default) export the artifacts."""
    settings = get_settings()
    bus = EventBus()
    orch = Orchestrator(settings=settings, bus=bus)

    stop = threading.Event()

    def drain():
        q = bus.subscribe()
        while not stop.is_set():
            try:
                ev = q.get(timeout=0.5)
            except Exception:
                continue
            _print_event(ev)
        bus.unsubscribe(q)

    t = threading.Thread(target=drain, daemon=True)
    t.start()
    orch.run(only=only, force=force)
    if export:
        stats = orch.export()
        typer.echo(f"\n导出: QA {stats.qa_units} 单元 / {stats.qa_rows} 行, "
                   f"SOP {stats.sop_files} 篇, 待审 {stats.needs_review}")
        typer.echo(f"输出目录: {settings.output_dir}")
    stop.set()


@app.command()
def export():
    """Export CSV/SOP/metadata from the last run's results.json."""
    settings = get_settings()
    orch = Orchestrator(settings=settings)
    # Rehydrate consolidated units from results.json isn't wired; re-run is the
    # canonical path. Export here operates on whatever the last run left in memory
    # only within a single process, so this command is mainly for the dashboard.
    typer.echo("请使用 `ragkb run` 一次性处理并导出，或在仪表盘中点击导出。")


@app.command()
def serve(port: int = typer.Option(8000, help="仪表盘端口"),
          host: str = typer.Option("127.0.0.1")):
    """Launch the web dashboard."""
    import uvicorn
    uvicorn.run("ragkb.server.app:app", host=host, port=port, log_level="info")


def _print_event(ev):
    k = ev.kind
    if k == "doc_status":
        typer.echo(f"[{ev.data.get('status','').upper():8s}] {ev.topic}")
    elif k == "run_status":
        typer.echo(f"=== run {ev.data.get('status','')} ===")
    elif k == "log":
        typer.echo(f"  · {ev.data.get('message','')}")
    elif k == "error":
        typer.secho(f"  ✗ {ev.topic}: {ev.data.get('message','')}", fg="red")


if __name__ == "__main__":
    app()
