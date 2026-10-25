"""EX by BIRD difficulty for the M1b dev rows, and the 40 seeded m1b_mix failures labelled in 003."""

import json
import random
from collections import Counter
from pathlib import Path

from sqltune.config.models import DataConfig
from sqltune.data.bird import load_bird

RUNS = [
    "m1_base__zero__birddiag",
    "m1_base__rows3retry__birddiag",
    "m1_m0_qlora__rows3retry__birddiag",
    "m1_m1b_mix__zero__birddiag",
    "m1_m1b_mix__rows3retry__birddiag",
    "m1_qwen7b__retry__birddiag",
    "m1_qwen14b__retry__birddiag",
]


def rows(run: str) -> list[dict]:
    path = Path("results") / run / "predictions.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


print("| run | simple | moderate | challenging |")
print("|---|---|---|---|")
for run in RUNS:
    if not (Path("results") / run).exists():
        continue
    by = {}
    for r in rows(run):
        if r["outcome"] == "gold_error":
            continue
        by.setdefault(r["difficulty"], []).append(r["col_tolerant_correct"])
    cells = [
        f"{100 * sum(v) / len(v):.1f} (n={len(v)})"
        for v in (by.get(d, [0]) for d in ("simple", "moderate", "challenging"))
    ]
    print(f"| {run} | " + " | ".join(cells) + " |")

evidence = {e.id: e.evidence for e in load_bird(DataConfig(bird_split="diag"))}
fails = [
    r
    for r in rows("m1_m1b_mix__rows3retry__birddiag")
    if not r["col_tolerant_correct"] and r["outcome"] != "gold_error"
]
print("\nfailures:", len(fails), Counter(r["outcome"] for r in fails))
pick = random.Random(0).sample(fails, 40)
for i, r in enumerate(sorted(pick, key=lambda r: r["id"])):
    print(f"\n### {i + 1} {r['id']} {r['db_id']} {r['difficulty']} {r['outcome']}")
    print("Q:", r["question"])
    print("E:", evidence.get(r["id"]))
    print("GOLD:", r["gold_sql"])
    print("PRED:", r["pred_sql"])
    if r["pred_error"]:
        print("ERR:", r["pred_error"][:200])
