"""Plot the README training figures from MLflow and the eval results.

Purpose: Show what the experiment tracking recorded without needing `mlflow ui`: the M1b
loss curves and the M1a learning curve (how EX grows with Spider training data).
Flow: MLflow (`get_metric_history` of `loss` / `eval_loss`) + results/m1_*/metrics.json ->
docs/img/m1b_loss.png, docs/img/m1_learning_curve.png.
Params: LOSS_RUNS (training run names; the latest FINISHED run of each name is used) and
LEARNING_CURVE (adapter name -> share of Spider train) below.
Try: `uv run python scripts/plot_training_curves.py`.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
from mlflow.tracking import MlflowClient

TRACKING_URI = "sqlite:///mlflow.db"
OUT = Path("docs/img")
LOSS_RUNS = ["m1b_mix", "m0_qlora"]
LEARNING_CURVE = {"lc_f0125": 0.125, "lc_f025": 0.25, "lc_f05": 0.5, "lc_f1": 1.0}
# categorical slots 1 and 2 of the reference palette (blue, orange); text stays neutral
BLUE, ORANGE, INK, MUTED = "#2a78d6", "#eb6834", "#1f1f1e", "#6b6a64"


def _run_id(client: MlflowClient, name: str) -> str:
    runs = mlflow.search_runs(
        search_all_experiments=True,
        filter_string=f"tags.mlflow.runName = '{name}' and attributes.status = 'FINISHED'",
        order_by=["attributes.start_time DESC"],
        output_format="list",
    )
    if not runs:
        raise SystemExit(f"no finished MLflow run named {name}")
    return runs[0].info.run_id


def _history(client: MlflowClient, run_id: str, key: str) -> tuple[list[int], list[float]]:
    points = sorted(client.get_metric_history(run_id, key), key=lambda m: m.step)
    return [m.step for m in points], [m.value for m in points]


def _style(ax: plt.Axes) -> None:
    ax.grid(axis="y", color="#e4e3df", linewidth=0.8)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=MUTED)


def plot_loss(client: MlflowClient) -> Path:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    mix = _run_id(client, "m1b_mix")
    steps, loss = _history(client, mix, "loss")
    ax.plot(steps, loss, color=BLUE, linewidth=2, label="m1b_mix train loss (Spider + BIRD)")
    steps, eval_loss = _history(client, mix, "eval_loss")
    ax.plot(
        steps,
        eval_loss,
        color=BLUE,
        linewidth=2,
        linestyle="--",
        marker="o",
        markersize=5,
        label="m1b_mix validation loss (unseen databases)",
    )
    steps, loss = _history(client, _run_id(client, "m0_qlora"), "loss")
    ax.plot(steps, loss, color=ORANGE, linewidth=2, label="M0 train loss (Spider only)")
    ax.set_xlabel("optimizer step", color=INK)
    ax.set_ylabel("loss (SQL tokens only)", color=INK)
    ax.set_title("Training loss", color=INK, loc="left")
    _style(ax)
    ax.legend(frameon=False)
    fig.tight_layout()
    path = OUT / "m1b_loss.png"
    fig.savefig(path)
    return path


def _ex(run: str) -> float:
    metrics = json.loads((Path("results") / run / "metrics.json").read_text(encoding="utf-8"))
    return 100 * metrics["ex_col_tolerant"]


def plot_learning_curve() -> Path:
    fig, ax = plt.subplots(figsize=(7, 4), dpi=150)
    shares = list(LEARNING_CURVE.values())
    for set_name, label, color in (
        ("spider", "Spider dev", BLUE),
        ("birddiag", "bird_diag", ORANGE),
    ):
        ex = [_ex(f"m1_{name}__zero__{set_name}") for name in LEARNING_CURVE]
        ax.plot(shares, ex, color=color, linewidth=2, marker="o", markersize=6)
        base = _ex(f"m1_base__zero__{set_name}")
        ax.axhline(base, color=color, linewidth=1, linestyle=":")
        ax.annotate(
            f"{label}",
            (shares[-1], ex[-1]),
            xytext=(8, 0),
            textcoords="offset points",
            va="center",
            color=INK,
        )
        ax.annotate(
            f"base 3B ({base:.1f})",
            (shares[0], base),
            xytext=(0, 4),
            textcoords="offset points",
            color=MUTED,
            fontsize=8,
        )
    ax.set_xscale("log", base=2)
    ax.set_xticks(shares, ["1/8", "1/4", "1/2", "all (7k)"])
    ax.set_xlabel("share of Spider train used (Spider-only adapters)", color=INK)
    ax.set_ylabel("EX % (zero-shot)", color=INK)
    ax.set_title("More Spider data helps Spider, not BIRD", color=INK, loc="left")
    _style(ax)
    fig.tight_layout()
    path = OUT / "m1_learning_curve.png"
    fig.savefig(path)
    return path


def main() -> None:
    mlflow.set_tracking_uri(TRACKING_URI)
    OUT.mkdir(parents=True, exist_ok=True)
    client = MlflowClient()
    for path in (plot_loss(client), plot_learning_curve()):
        print(f"wrote {path}")


if __name__ == "__main__":
    main()
