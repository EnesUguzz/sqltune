"""Log a finished run to MLflow (params, metrics, and the run folder as artifacts).

Purpose: MLflow answers "which config was better?" across many runs with sortable
tables and charts; the Markdown report answers "why?" for one run.
Flow: runner / trainer -> log_to_mlflow() -> MLflow (SQLite) -> `uv run mlflow ui`.
Params: EvalConfig.mlflow_tracking_uri, RunConfig.experiment_name, RunConfig.run_name.
Try: `uv run mlflow ui --backend-store-uri sqlite:///mlflow.db`, select runs, "Compare".
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from sqltune.config.models import AppConfig

MAX_PARAM_CHARS = 500


def flatten(data: dict[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in data.items():
        name = f"{prefix}{key}"
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{name}."))
        else:
            out[name] = value
    return out


def log_to_mlflow(cfg: AppConfig, metrics: dict[str, float], run_dir: Path) -> str:
    import mlflow  # heavy import, only when a run is logged

    mlflow.set_tracking_uri(cfg.evaluation.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run.experiment_name)
    params = flatten(cfg.model_dump(mode="json"))
    with mlflow.start_run(run_name=cfg.run.run_name) as run:
        mlflow.log_params({k: str(v)[:MAX_PARAM_CHARS] for k, v in params.items()})
        mlflow.log_metrics(metrics)
        mlflow.log_artifacts(str(run_dir))
        return str(run.info.run_id)
