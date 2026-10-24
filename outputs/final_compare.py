"""One-off: paired McNemar for the test-once rows (tolerant EX), plus strict EX and n."""

import json
from pathlib import Path

from sqltune.evaluation.metrics import mcnemar_exact_p


def load(run: str) -> dict[str, dict]:
    path = Path("results") / run / "predictions.jsonl"
    return {r["id"]: r for r in map(json.loads, path.read_text().splitlines())}


PAIRS = [
    ("m1b_mix__zero", "base__zero"),
    ("m1b_mix__rows3retry", "base__rows3retry"),
    ("m1b_mix__zero", "m0_qlora__zero"),
    ("m1b_mix__rows3retry", "m0_qlora__rows3retry"),
    ("m1b_mix__rows3retry", "qwen7b__rows3retry"),
    ("m1b_mix__zero", "qwen7b__zero"),
    ("m1b_mix__rows3retry", "qwen14b__rows3retry"),
]
for s in ("birdholdout", "spidertest"):
    for a, b in PAIRS:
        ra, rb = load(f"m1_{a}__{s}"), load(f"m1_{b}__{s}")
        ids = [i for i in ra if ra[i]["outcome"] != "gold_error"]
        fixed = sum(ra[i]["col_tolerant_correct"] and not rb[i]["col_tolerant_correct"] for i in ids)
        broken = sum(rb[i]["col_tolerant_correct"] and not ra[i]["col_tolerant_correct"] for i in ids)
        d = 100 * (fixed - broken) / len(ids)
        print(f"{s:12} {a:22} vs {b:22} d={d:+5.1f} fixed/broken {fixed}/{broken} p={mcnemar_exact_p(fixed, broken):.2g}")
