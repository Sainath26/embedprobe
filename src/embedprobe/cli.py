"""Command-line interface: ``embedprobe run`` and ``embedprobe compare``."""

from __future__ import annotations

import json
from pathlib import Path
from typing import List, Optional

from rich.console import Console
import typer

app = typer.Typer(help="Diagnose multilingual embedding models on your own data.")
console = Console()


@app.command()
def run(
    models: List[str] = typer.Option(..., "--models", "-m", help="Hub model name (repeatable)."),
    data: Path = typer.Option(..., "--data", "-d", help="CSV/Parquet with parallel pairs."),
    src_col: str = typer.Option(..., "--src", "--src-col", help="Source-language text column."),
    tgt_col: str = typer.Option(..., "--tgt", "--tgt-col", help="Target-language text column."),
    topic_col: str = typer.Option("topic", help="Topic column (Level 2 is skipped if absent)."),
    out: Path = typer.Option(Path("embedprobe_report"), "--out", "-o", help="Output basename."),
    seed: int = typer.Option(42, help="Random seed."),
    batch_size: int = typer.Option(32, help="Encoding batch size."),
    device: Optional[str] = typer.Option(None, help="Torch device, e.g. cuda."),
    max_pairs: Optional[int] = typer.Option(None, help="Subsample the dataset to this many pairs."),
    umap: bool = typer.Option(False, "--umap", help="Compute UMAP projections (needs umap-learn)."),
):
    """Run the four-level diagnostic and write <out>.json and <out>.html."""
    from embedprobe.probe import probe

    model_names = _normalize_models(models)
    report = probe(
        models=model_names, data=data, src_col=src_col, tgt_col=tgt_col, topic_col=topic_col,
        seed=seed, batch_size=batch_size, device=device, max_pairs=max_pairs,
        compute_umap=umap,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    json_path, html_path = out.with_suffix(".json"), out.with_suffix(".html")
    report.to_json(json_path)
    report.to_html(html_path)
    console.print(report.summary().round(4).to_string())
    console.print(f"\n[green]Wrote[/green] {json_path} and {html_path}")


@app.command()
def compare(
    reports: List[Path] = typer.Argument(..., help="embedprobe JSON reports to merge."),
    out: Optional[Path] = typer.Option(None, "--out", "-o", help="Write merged HTML here."),
):
    """Merge previously saved JSON reports into one comparison."""
    from embedprobe.report import ModelDiagnostics, ProbeReport

    diagnostics, run_meta = [], {}
    for path in reports:
        payload = json.loads(path.read_text(encoding="utf-8"))
        run_meta = run_meta or payload.get("run", {})
        for entry in payload.get("models", []):
            diagnostics.append(
                ModelDiagnostics(
                    model_name=entry.get("model", path.stem),
                    meta=entry.get("meta", {}),
                    level0=entry.get("level0"),
                    level1=entry.get("level1"),
                    level2=entry.get("level2"),
                    level3=entry.get("level3"),
                )
            )
    merged = ProbeReport(models=diagnostics, run_meta=run_meta)
    console.print(merged.summary().round(4).to_string())
    if out is not None:
        merged.to_html(out)
        console.print(f"\n[green]Wrote[/green] {out}")


def _normalize_models(models: List[str]) -> List[str]:
    normalized = []
    for item in models:
        normalized.extend(part.strip() for part in item.split(",") if part.strip())
    return normalized


if __name__ == "__main__":
    app()
