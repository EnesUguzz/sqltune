"""One comparison table over every M1 run (results/m1_<model>__<variant>__<set>/).

Purpose: The matrix has dozens of runs; this puts them side by side with the delta against
the base model's zero-shot run on the same set (with a database-cluster 95% CI), the share of
invented-column errors, retries and the training cost of each adapter.
Flow: results/m1_*/{metrics.json, predictions.jsonl, config.yaml} (+ results/<adapter>/
train_summary.json) -> build_summary() -> results/SUMMARY.md.
Params: REFERENCE_MODEL / REFERENCE_VARIANT (the row every delta is taken against).
Try: `make summarize` and open results/SUMMARY.md.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from sqltune.evaluation.metrics import db_cluster_bootstrap_ci
from sqltune.evaluation.report import error_kind, load_rows

PREFIX = "m1_"
REFERENCE_MODEL = "base"
REFERENCE_VARIANT = "zero"


def _runs(results_dir: Path) -> list[dict[str, Any]]:
    runs = []
    for run_dir in sorted(results_dir.glob(f"{PREFIX}*__*__*")):
        if not (run_dir / "metrics.json").exists():
            continue
        model, variant, eval_set = run_dir.name[len(PREFIX) :].split("__")
        runs.append(
            {
                "model": model,
                "variant": variant,
                "set": eval_set,
                "metrics": json.loads((run_dir / "metrics.json").read_text(encoding="utf-8")),
                "rows": load_rows(run_dir),
                "train": _train_summary(results_dir / model),
            }
        )
    return runs


def _train_summary(path: Path) -> dict[str, Any] | None:
    file = path / "train_summary.json"
    return json.loads(file.read_text(encoding="utf-8")) if file.exists() else None


def _pct(value: float) -> str:
    return f"{100 * value:.1f}"


def build_summary(results_dir: Path) -> str:
    runs = _runs(results_dir)
    refs = {
        r["set"]: r
        for r in runs
        if (r["model"], r["variant"]) == (REFERENCE_MODEL, REFERENCE_VARIANT)
    }
    lines = [
        "# M1 results",
        "",
        "EX = execution accuracy, column-order tolerant (strict in brackets). Δ is against "
        f"`{REFERENCE_MODEL}` / `{REFERENCE_VARIANT}` on the same set, with a 95% CI from "
        "resampling databases. nsc = share of questions failing with `no such column` "
        "(invented columns).",
        "",
        "| model | variant | set | n | EX (strict) | Δ EX [95% CI] | exec_error | nsc | retried "
        "| p50 ms | train min | train GiB |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for set_name in sorted({r["set"] for r in runs}, key=lambda s: (s != "spider", s)):
        set_runs = [r for r in runs if r["set"] == set_name]
        ref = refs.get(set_name)
        for r in sorted(set_runs, key=lambda r: -r["metrics"]["ex_col_tolerant"]):
            m, rows = r["metrics"], r["rows"]
            if ref is None or r is ref:
                delta = "—"
            else:
                lo, hi = db_cluster_bootstrap_ci(ref["rows"], rows)
                d = m["ex_col_tolerant"] - ref["metrics"]["ex_col_tolerant"]
                delta = f"{100 * d:+.1f} [{100 * lo:+.1f}, {100 * hi:+.1f}]"
            nsc = sum(
                1
                for x in rows
                if x["outcome"] == "exec_error"
                and error_kind(x.get("pred_error")) == "no such column"
            )
            train = r["train"]
            train_cols = (
                f"{train['runtime_s'] / 60:.0f} | {train['peak_vram_gib']:.1f}"
                if train
                else "— | —"
            )
            lines.append(
                f"| {r['model']} | {r['variant']} | {set_name} | {m['n']:.0f} "
                f"| {_pct(m['ex_col_tolerant'])} ({_pct(m['ex'])}) | {delta} "
                f"| {_pct(m.get('rate_exec_error', 0.0))} | {_pct(nsc / max(len(rows), 1))} "
                f"| {sum(1 for x in rows if x.get('retried'))} "
                f"| {m.get('latency_p50_ms', 0.0):.0f} | {train_cols} |"
            )
        transport = sum(r["metrics"].get("n_transport_error", 0) for r in set_runs)
        if transport:
            lines.append(
                f"\n**{transport:.0f} transport errors on {set_name}; rerun those runs.**\n"
            )
    return "\n".join(lines) + "\n"
