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
