"""Write a run's Markdown report, optionally against a reference run.

Purpose: MLflow says which run scored higher; the report says why: which outcome types
moved, which questions were fixed or broken, and whether the change is beyond noise.
Flow: runner (rows, metrics) + results/<reference_run>/ -> render_report() -> report.md.
Params: RunConfig.reference_run; EXAMPLES_PER_SIDE examples of fixed and broken questions.
Try: `make eval CONFIG=experiments/configs/m0/qlora_eval.yaml`, open results/<run>/report.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqltune.evaluation.metrics import mcnemar_exact_p

EXAMPLES_PER_SIDE = 5
PERCENT_METRICS = ("ex", "ex_col_tolerant")


def load_rows(run_dir: Path) -> list[dict[str, Any]]:
    lines = (run_dir / "predictions.jsonl").read_text(encoding="utf-8").splitlines()
    return [json.loads(line) for line in lines if line.strip()]


def _fmt(key: str, value: float) -> str:
    if key in PERCENT_METRICS or key.startswith("rate_"):
        return f"{100 * value:.1f}"
    return f"{value:.1f}" if isinstance(value, float) else str(value)


def _example_block(title: str, rows: list[dict[str, Any]]) -> list[str]:
    lines = [f"### {title}", ""]
    for r in rows[:EXAMPLES_PER_SIDE]:
        lines += [
            f"- **{r['id']}** ({r['db_id']}, now `{r['outcome']}`): {r['question']}",
            f"  - gold: `{r['gold_sql']}`",
            f"  - pred: `{r['pred_sql']}`",
        ]
    return [*lines, ""]


def render_report(
    run_name: str,
    rows: list[dict[str, Any]],
    metrics: dict[str, float],
    reference: tuple[str, list[dict[str, Any]], dict[str, float]] | None = None,
) -> str:
    lines = [f"# Run `{run_name}`", ""]
    if reference is None:
        lines += ["| metric | value |", "|---|---|"]
        lines += [f"| {k} | {_fmt(k, v)} |" for k, v in metrics.items()]
        return "\n".join(lines) + "\n"

    ref_name, ref_rows, ref_metrics = reference
    lines += [f"| metric | {ref_name} | {run_name} | Δ vs {ref_name} |", "|---|---|---|---|"]
    for key, value in metrics.items():
        if key in ref_metrics:
            delta = value - ref_metrics[key]
            sign = "+" if delta >= 0 else ""
            lines.append(
                f"| {key} | {_fmt(key, ref_metrics[key])} | {_fmt(key, value)} "
                f"| {sign}{_fmt(key, delta)} |"
            )
    ref_by_id = {r["id"]: r for r in ref_rows}
    paired = [
        (ref_by_id[r["id"]], r)
        for r in rows
        if r["id"] in ref_by_id
        and "gold_error" not in (r["outcome"], ref_by_id[r["id"]]["outcome"])
    ]
    fixed = [
        new for old, new in paired if old["outcome"] != "correct" and new["outcome"] == "correct"
    ]
    broken = [
        new for old, new in paired if old["outcome"] == "correct" and new["outcome"] != "correct"
    ]
    p = mcnemar_exact_p(len(fixed), len(broken))
    lines += [
        "",
        f"Paired questions: {len(paired)}; fixed: {len(fixed)}; broken: {len(broken)}; "
        f"exact McNemar p = {p:.4g}",
        "",
    ]
    lines += _example_block("Fixed (wrong before, correct now)", fixed)
    lines += _example_block("Broken (correct before, wrong now)", broken)
    return "\n".join(lines) + "\n"
