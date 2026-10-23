"""Command-line entry point: `sqltune <command>`.

Purpose: One place that wires config files to the pipeline steps (env check, download,
eval, train), so every run is reproducible from a YAML file.
Flow: `sqltune <cmd> --config experiments/configs/....yaml` -> load_config -> step.
Params: --config for every pipeline command; see each command's --help.
Try: `uv run sqltune check-env`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from sqltune.config.models import TrainConfig

app = typer.Typer(no_args_is_help=True, add_completion=False)


@app.command("check-env")
def check_env(
    docker: Annotated[bool, typer.Option(help="Also test GPU access from Docker.")] = True,
) -> None:
    """Print torch / CUDA / bitsandbytes / Unsloth status and Docker GPU access."""
    from sqltune.gpu import docker_gpu_ok, gpu_report

    report = gpu_report()
    if docker:
        report["docker_gpu"] = docker_gpu_ok()
    width = max(len(k) for k in report)
    for key, value in report.items():
        typer.echo(f"{key:<{width}}  {value}")


@app.command("download")
def download(
    config: Annotated[Path, typer.Option(help="Experiment YAML.")] = Path(
        "experiments/configs/base.yaml"
    ),
) -> None:
    """Download and unpack Spider (questions + SQLite databases) into <data_dir>/spider/."""
    from sqltune.config.loader import load_config
    from sqltune.data.download import download_spider
    from sqltune.data.spider import load_spider

    cfg = load_config(config)
    root = download_spider(cfg.data)
    for split in ("train", "dev"):
        typer.echo(f"{split}: {len(load_spider(cfg.data, split))} questions")
    typer.echo(f"databases: {len(list((root / 'database').iterdir()))} in {root}")

    from sqltune.data.bird import load_bird, split_bird
    from sqltune.data.download import download_bird

    bird_root = download_bird(cfg.data)
    examples = load_bird(cfg.data.model_copy(update={"bird_split": "all"}))
    diag, holdout = split_bird(examples, cfg.data.seed)
    typer.echo(
        f"BIRD Mini-Dev: {len(examples)} questions in {bird_root}; "
        f"diag {len(diag)} ({len({e.db_id for e in diag})} DBs), "
        f"holdout {len(holdout)} ({len({e.db_id for e in holdout})} DBs)"
    )


@app.command("eval")
def eval_(
    config: Annotated[Path, typer.Option(help="Experiment YAML.")],
) -> None:
    """Generate SQL for every question with the configured model, score it, write the run."""
    import json

    from sqltune.config.loader import load_config
    from sqltune.runner import run_eval

    out = run_eval(load_config(config))
    metrics = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    if metrics["n_transport_error"]:
        typer.secho(
            f"WARNING: {metrics['n_transport_error']:.0f} requests failed after retries and "
            "were scored as no_sql; check the server and re-run before using this EX.",
            fg=typer.colors.RED,
        )
    typer.echo(
        f"{out}: EX {100 * metrics['ex']:.1f} (col-tolerant {100 * metrics['ex_col_tolerant']:.1f}), n={metrics['n']:.0f}"
    )


@app.command("train")
def train_(
    config: Annotated[Path, typer.Option(help="Experiment YAML.")],
) -> None:
    """Fine-tune the base model on Spider train with LoRA/QLoRA; saves adapters/<run_name>/."""
    from sqltune.config.loader import load_config
    from sqltune.training.train import train

    typer.echo(f"wrote {train(load_config(config))}")


@app.command("summarize")
def summarize(
    out: Annotated[Path, typer.Option(help="Markdown file to write.")] = Path("results/SUMMARY.md"),
    results_dir: Annotated[Path, typer.Option(help="Folder with run folders.")] = Path("results"),
) -> None:
    """Write one comparison table over every results/m1_* run."""
    from sqltune.evaluation.summary import build_summary

    out.write_text(build_summary(results_dir), encoding="utf-8")
    typer.echo(f"wrote {out}")


@app.command("dummy-adapter")
def dummy_adapter(
    out: Annotated[Path, typer.Option(help="Output folder.")] = Path("adapters/dummy"),
) -> None:
    """Save an untrained LoRA adapter (default TrainConfig shape) to check vLLM LoRA serving."""
    from sqltune.training.dummy_adapter import make_dummy_adapter

    path = make_dummy_adapter(TrainConfig(), out)
    typer.echo(f"wrote {path}")


if __name__ == "__main__":
    app()
