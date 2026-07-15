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
def export(with_paraphrase: bool = typer.Option(
        False, "--with-paraphrase",
        help="额外导出实验性扩写版；默认主产物仅包含原始主问题")):
    """Regenerate CSV/SOP/metadata/zip from the last run's results.json.

    Reproducible outside the producing process: rehydrates consolidated units
    (and the redaction map) from output/, then re-exports. Use this after hand-
    editing results.json, or to re-run export with updated packaging logic.

    `--with-paraphrase` writes a separate experimental artifact; the default
    production artifact contains only the source-faithful primary query."""
    settings = get_settings()
    from ragkb.pipeline.export import export_all, load_results
    qa, sop = load_results(settings.output_dir)
    qa = [u for u in qa if u.publication_status == "approved" and u.semantic_ok]
    stats = export_all(qa, sop, settings.output_dir,
                       include_paraphrases=with_paraphrase)
    variant = "含扩写实验版" if with_paraphrase else "忠实原文主版本"
    typer.echo(
        f"已导出（{variant}）：QA单元 {stats.qa_units}，检索行 {stats.qa_rows}，"
        f"SOP {stats.sop_files}，待复核 {stats.needs_review}，模块 {stats.modules}")


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
