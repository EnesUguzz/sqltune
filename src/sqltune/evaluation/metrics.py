"""Aggregate per-question outcomes into run metrics, and test paired differences.

Purpose: One number (EX) to compare runs, plus the outcome mix that explains it and a
significance test so a small delta is not over-read.
Flow: runner rows -> compute_metrics() -> metrics.json / MLflow / report.md.
Params: none. EX = correct / (n - gold_error); rate_* = share of scorable questions.
ex_col_tolerant counts a result with the right values in a different column order as correct.
Try: `mcnemar_exact_p(fixed=30, broken=12)` -> ~0.008: unlikely to be noise.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, get_args

import numpy as np

from sqltune.domain import Outcome

OUTCOMES: tuple[str, ...] = get_args(Outcome)


def compute_metrics(rows: list[dict[str, Any]]) -> dict[str, float]:
    counts = Counter(r["outcome"] for r in rows)
    n = len(rows)
    scorable = n - counts["gold_error"]
    denom = max(scorable, 1)
    metrics: dict[str, float] = {
        "n": n,
        "n_gold_error": counts["gold_error"],
        "ex": counts["correct"] / denom,
        "ex_col_tolerant": sum(
            1 for r in rows if r["outcome"] != "gold_error" and r["col_tolerant_correct"]
        )
        / denom,
    }
    for outcome in OUTCOMES:
        if outcome != "gold_error":
            metrics[f"rate_{outcome}"] = counts[outcome] / denom
    latencies = [r["latency_ms"] for r in rows if r.get("latency_ms") is not None]
    if latencies:
        metrics["latency_p50_ms"] = float(np.percentile(latencies, 50))
        metrics["latency_p95_ms"] = float(np.percentile(latencies, 95))
    for key in ("prompt_tokens", "completion_tokens"):
        values = [r[key] for r in rows if r.get(key) is not None]
        if values:
            metrics[f"mean_{key}"] = float(np.mean(values))
    return metrics


def mcnemar_exact_p(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)
