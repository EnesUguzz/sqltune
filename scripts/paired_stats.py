"""Paired statistics behind the tables in docs/decisions/003 and 004.

Purpose: Every comparison in the decision notes is recomputed here from
results/<run>/predictions.jsonl, so each number in the docs can be checked.
Flow: run pairs per group -> per-question paired deltas (column-order-tolerant EX, gold
errors excluded) -> Δ, fixed / broken, exact McNemar p, Holm-adjusted p within the group,
question-level 95% CI and database-cluster bootstrap 95% CI -> markdown rows on stdout.
The `did` group compares two fine-tune gains (difference in differences), and `perdb`
prints EX per database.
Params: group names on the command line (default: all of test, dev7b, sqlite_rules, indist, did, perdb).
Holm adjusts over the comparisons of one group only. Both CIs are rough with 5-6 databases.
Try: `uv run python scripts/paired_stats.py test`.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path

from sqltune.evaluation.metrics import db_cluster_bootstrap_ci, mcnemar_exact_p

RESULTS = Path("results")

GROUPS: dict[str, list[tuple[str, str, str]]] = {  # (set, run a, run b): a - b
    "test": [
        ("birdholdout", "m1b_mix__zero", "base__zero"),
        ("birdholdout", "m1b_mix__rows3retry", "base__rows3retry"),
        ("birdholdout", "m1b_mix__rows3retry", "m0_qlora__rows3retry"),
        ("birdholdout", "m1b_mix__rows3retry", "qwen7b__rows3retry"),
        ("birdholdout", "m1b_mix__rows3retry", "qwen14b__rows3retry"),
        ("spidertest", "m1b_mix__zero", "base__zero"),
        ("spidertest", "m1b_mix__rows3retry", "m0_qlora__rows3retry"),
        ("spidertest", "m1b_mix__rows3retry", "qwen7b__rows3retry"),
    ],
    "dev7b": [
        ("birddiag", "m1b_mix7b__zero", "qwen7b_own__zero"),
        ("birddiag", "m1b_mix7b__rows3retry", "qwen7b_own__rows3retry"),
        ("birddiag", "m1b_mix7b_fp8__zero", "qwen7b_fp8__zero"),
        ("birddiag", "m1b_mix7b_fp8__rows3retry", "qwen7b_fp8__rows3retry"),
        ("spider", "m1b_mix7b__zero", "qwen7b_own__zero"),
        ("spider", "m1b_mix7b_fp8__zero", "qwen7b_fp8__zero"),
        ("spider", "m1b_mix7b_fp8__rows3retry", "qwen7b_fp8__rows3retry"),
        ("birddiag", "qwen7b_own__rows3retry", "qwen7b__rows3retry"),
        ("birddiag", "qwen7b_fp8__zero", "qwen7b_own__zero"),
        ("birddiag", "m1b_mix__rows3retry", "qwen7b__rows3retry"),
        ("birddiag", "m1b_mix__rows3retry", "qwen7b_fp8__rows3retry"),
        ("spider", "m1b_mix__rows3retry", "qwen7b_fp8__rows3retry"),
    ],
    "sqlite_rules": [
        ("birddiag", "base_sqliterules__zero", "base__zero"),
        ("birddiag", "base_sqliterules__rows3retry", "base__rows3retry"),
        ("birddiag", "m1b_mix__zero", "base_sqliterules__zero"),
        ("birddiag", "m1b_mix__rows3retry", "base_sqliterules__rows3retry"),
        ("spider", "base_sqliterules__zero", "base__zero"),
        ("spider", "base_sqliterules__rows3retry", "base__rows3retry"),
        ("spider", "m1b_mix__rows3retry", "base_sqliterules__rows3retry"),
    ],
    "indist": [
        ("birdval", "m1b_mix__zero", "base__zero"),
        ("birdtrain", "m1b_mix__zero", "base__zero"),
        ("birdval", "m1b_mix7b_fp8__zero", "qwen7b_fp8__zero"),
        ("birdtrain", "m1b_mix7b_fp8__zero", "qwen7b_fp8__zero"),
    ],
}
# 3B gain minus 7B gain on the same questions: ((a1 - b1) - (a2 - b2))
DID = [
    ("birddiag", ("m1b_mix__zero", "base__zero"), ("m1b_mix7b_fp8__zero", "qwen7b_fp8__zero")),
    (
        "birddiag",
        ("m1b_mix__rows3retry", "base__rows3retry"),
        ("m1b_mix7b_fp8__rows3retry", "qwen7b_fp8__rows3retry"),
    ),
    (
        "birddiag",
        ("m1b_mix__rows3retry", "base__rows3retry"),
        ("m1b_mix7b__rows3retry", "qwen7b_own__rows3retry"),
    ),
]
PERDB = [
    "birdholdout",
    ["base__rows3retry", "m1b_mix__rows3retry", "qwen7b__rows3retry", "qwen14b__rows3retry"],
]


def load(run: str, split: str) -> list[dict] | None:
    path = RESULTS / f"m1_{run}__{split}" / "predictions.jsonl"
    if not path.exists():
        return None
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def correct(rows: list[dict]) -> dict[str, int]:
    return {
        r["id"]: int(bool(r["col_tolerant_correct"])) for r in rows if r["outcome"] != "gold_error"
    }


def holm(ps: list[float]) -> list[float]:
    order = sorted(range(len(ps)), key=lambda i: ps[i])
    out, running = [0.0] * len(ps), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(ps) - rank) * ps[i]))
        out[i] = running
    return out


def mean_ci(d: list[int]) -> tuple[float, float, float]:
    n = len(d)
    m = sum(d) / n
    sd = math.sqrt(sum((x - m) ** 2 for x in d) / (n - 1))
    return 100 * m, 100 * (m - 1.96 * sd / math.sqrt(n)), 100 * (m + 1.96 * sd / math.sqrt(n))


def pairs(group: str) -> None:
    lines, ps = [], []
    for split, a, b in GROUPS[group]:
        ra, rb = load(a, split), load(b, split)
        if ra is None or rb is None:
            continue
        ca, cb = correct(ra), correct(rb)
        ids = sorted(ca.keys() & cb.keys())
        fixed = sum(ca[i] and not cb[i] for i in ids)
        broken = sum(cb[i] and not ca[i] for i in ids)
        d, lo, hi = mean_ci([ca[i] - cb[i] for i in ids])
        blo, bhi = db_cluster_bootstrap_ci(rb, ra)
        p = mcnemar_exact_p(fixed, broken)
        ps.append(p)
        lines.append(
            [
                split,
                f"{a} vs {b}",
                f"{d:+.1f}",
                f"{fixed} / {broken}",
                f"{p:.2g}",
                f"[{lo:+.1f}, {hi:+.1f}]",
                f"[{100 * blo:+.1f}, {100 * bhi:+.1f}]",
                str(len(ids)),
            ]
        )
    if not lines:
        return
    print(f"\n## {group}\n")
    print(
        "| set | comparison | Δ | fixed / broken | p | p Holm | 95% CI (questions) | 95% CI (databases) | n |"
    )
    print("|---|---|---|---|---|---|---|---|---|")
    for row, ph in zip(lines, holm(ps), strict=True):
        print("| " + " | ".join([*row[:5], f"{ph:.2g}", *row[5:]]) + " |")


def did() -> None:
    print("\n## did (3B gain minus 7B gain, same questions)\n")
    print("| set | 3B pair | 7B pair | 3B gain | 7B gain | difference | 95% CI (questions) |")
    print("|---|---|---|---|---|---|---|")
    for split, (a1, b1), (a2, b2) in DID:
        runs = [load(r, split) for r in (a1, b1, a2, b2)]
        if any(r is None for r in runs):
            continue
        c = [correct(r) for r in runs]  # type: ignore[arg-type]
        ids = sorted(set.intersection(*(set(x) for x in c)))
        g3 = [c[0][i] - c[1][i] for i in ids]
        g7 = [c[2][i] - c[3][i] for i in ids]
        d, lo, hi = mean_ci([x - y for x, y in zip(g3, g7, strict=True)])
        print(
            f"| {split} | {a1} - {b1} | {a2} - {b2} | {100 * sum(g3) / len(ids):+.1f} | "
            f"{100 * sum(g7) / len(ids):+.1f} | {d:+.1f} | [{lo:+.1f}, {hi:+.1f}] |"
        )


def perdb() -> None:
    split, runs = PERDB
    loaded = {r: load(r, split) for r in runs}
    if any(v is None for v in loaded.values()):
        return
    print(f"\n## perdb ({split}, EX % per database)\n")
    print("| database | n | " + " | ".join(runs) + " |")
    print("|---|---|" + "---|" * len(runs))
    dbs = sorted({r["db_id"] for r in loaded[runs[0]]})  # type: ignore[union-attr]
    for db in dbs:
        cells = []
        for r in runs:
            rows = [x for x in loaded[r] if x["db_id"] == db and x["outcome"] != "gold_error"]  # type: ignore[union-attr]
            cells.append(f"{100 * sum(x['col_tolerant_correct'] for x in rows) / len(rows):.1f}")
        print(f"| {db} | {len(rows)} | " + " | ".join(cells) + " |")


def main() -> None:
    wanted = sys.argv[1:] or [*GROUPS, "did", "perdb"]
    for g in wanted:
        if g == "did":
            did()
        elif g == "perdb":
            perdb()
        else:
            pairs(g)


if __name__ == "__main__":
    main()
