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
        # requests that failed after retries (server down, timeouts); scored as no_sql, so a
        # non-zero value means the EX of this run understates the model
        "n_transport_error": sum(1 for r in rows if r.get("error")),
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


def db_cluster_bootstrap_ci(
    ref_rows: list[dict[str, Any]],
    rows: list[dict[str, Any]],
    n: int = 2000,
    seed: int = 13,
    key: str = "col_tolerant_correct",
) -> tuple[float, float]:
    """95% CI of the paired EX delta, resampling whole databases with replacement.

    Questions on one database (and Spider's paraphrase pairs) are not independent, so
    resampling databases gives an honest interval where McNemar's p is overconfident."""
    ref = {r["id"]: r for r in ref_rows}
    by_db: dict[str, list[int]] = {}
    for r in rows:
        old = ref.get(r["id"])
        if old is None or "gold_error" in (r.get("outcome"), old.get("outcome")):
            continue
        by_db.setdefault(r["db_id"], []).append(int(bool(r[key])) - int(bool(old[key])))
    dbs = sorted(by_db)
    if not dbs:
        return 0.0, 0.0
    rng = np.random.default_rng(seed)
    deltas = []
    for _ in range(n):
        picked = [x for i in rng.integers(0, len(dbs), len(dbs)) for x in by_db[dbs[i]]]
        deltas.append(sum(picked) / len(picked))
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return float(lo), float(hi)


def mcnemar_exact_p(fixed: int, broken: int) -> float:
    """Two-sided exact McNemar test on the discordant pairs (binomial, p = 0.5)."""
    n = fixed + broken
    if n == 0:
        return 1.0
    k = min(fixed, broken)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / (1 << n)
    return min(1.0, 2 * tail)
