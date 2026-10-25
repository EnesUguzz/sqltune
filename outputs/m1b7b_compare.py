"""One-off: dev EX and paired McNemar for the 7B fine-tune (tolerant EX)."""

import json
from pathlib import Path

from sqltune.evaluation.metrics import mcnemar_exact_p


def load(run: str) -> dict[str, dict]:
    path = Path("results") / run / "predictions.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


def ex(rows: dict[str, dict]) -> tuple[float, float]:
    ids = [i for i in rows if rows[i]["outcome"] != "gold_error"]
    tol = 100 * sum(rows[i]["col_tolerant_correct"] for i in ids) / len(ids)
    strict = 100 * sum(rows[i]["outcome"] == "correct" for i in ids) / len(ids)
    return tol, strict


RUNS = [
    "qwen7b_own__zero",
    "qwen7b_own__rows3retry",
    "qwen7b__zero",
    "qwen7b__rows3retry",
    "m1b_mix7b__zero",
    "m1b_mix7b__rows3retry",
    "m1b_mix__zero",
    "m1b_mix__rows3retry",
    "qwen14b__retry",
    "qwen7b_fp8__zero",
    "qwen7b_fp8__rows3retry",
    "m1b_mix7b_fp8__zero",
    "m1b_mix7b_fp8__rows3retry",
]
PAIRS = [
    ("m1b_mix7b__zero", "qwen7b_own__zero"),
    ("m1b_mix7b__rows3retry", "qwen7b_own__rows3retry"),
    ("qwen7b_own__zero", "qwen7b__zero"),
    ("qwen7b_own__rows3retry", "qwen7b__rows3retry"),
    ("m1b_mix__rows3retry", "qwen7b__rows3retry"),
    ("m1b_mix7b__rows3retry", "m1b_mix__rows3retry"),
    ("m1b_mix7b__rows3retry", "qwen14b__retry"),
    ("m1b_mix7b_fp8__zero", "qwen7b_fp8__zero"),
    ("m1b_mix7b_fp8__rows3retry", "qwen7b_fp8__rows3retry"),
    ("m1b_mix7b_fp8__zero", "m1b_mix7b__zero"),
    ("qwen7b_fp8__zero", "qwen7b_own__zero"),
]
for s in ("birddiag", "spider"):
    for r in RUNS:
        p = Path("results") / f"m1_{r}__{s}"
        if p.exists():
            tol, strict = ex(load(p.name))
            print(f"{s:9} {r:24} EX {tol:5.1f} ({strict:5.1f})")
    for a, b in PAIRS:
        try:
            ra, rb = load(f"m1_{a}__{s}"), load(f"m1_{b}__{s}")
        except FileNotFoundError:
            continue
        ids = [i for i in ra if ra[i]["outcome"] != "gold_error"]
        fixed = sum(ra[i]["col_tolerant_correct"] and not rb[i]["col_tolerant_correct"] for i in ids)
        broken = sum(rb[i]["col_tolerant_correct"] and not ra[i]["col_tolerant_correct"] for i in ids)
        d = 100 * (fixed - broken) / len(ids)
        print(f"{s:9} {a:24} vs {b:24} d={d:+5.1f} {fixed}/{broken} p={mcnemar_exact_p(fixed, broken):.2g}")
