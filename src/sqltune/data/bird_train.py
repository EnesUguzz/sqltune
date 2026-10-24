"""Read BIRD train (SQLite) into Example records for fine-tuning.

Purpose: M1a showed a Spider-only fine-tune does not help on BIRD; BIRD train (9,428
questions on 69 databases, each with an evidence hint) is the in-distribution data. Its
databases are disjoint from Mini-Dev, so bird_diag / bird_holdout stay unseen.
Flow: <data_dir>/bird_train/{train.json, train_databases/<db>/<db>.sqlite} (+ optional
gold_failures.json from scripts/dataset_review.py) -> load_bird_train().
Params: DataConfig.bird_train_skip_failing_gold, bird_train_exclude_dbs.
Try: `len(load_bird_train(DataConfig()))`.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from sqltune.config.models import DataConfig
from sqltune.domain import Example

QUESTIONS_FILE = "train.json"
GOLD_FAILURES_FILE = "gold_failures.json"


def load_bird_train(cfg: DataConfig) -> list[Example]:
    root = cfg.data_dir / "bird_train"
    records = json.loads((root / QUESTIONS_FILE).read_text(encoding="utf-8"))
    skip: set[str] = set()
    failures = root / GOLD_FAILURES_FILE
    if cfg.bird_train_skip_failing_gold and failures.exists():
        skip = set(json.loads(failures.read_text(encoding="utf-8")))
    examples = [
        Example(
            id=f"birdtrain-{i:05d}",
            split="train",
            db_id=r["db_id"],
            db_path=root / "train_databases" / r["db_id"] / f"{r['db_id']}.sqlite",
            question=r["question"],
            gold_sql=r["SQL"],
            evidence=(r.get("evidence") or "").strip() or None,
        )
        for i, r in enumerate(records)
    ]
    excluded = set(cfg.bird_train_exclude_dbs)
    return [e for e in examples if e.id not in skip and e.db_id not in excluded]


def _tables(db_path: Path) -> set[str]:
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT name FROM sqlite_master WHERE type = 'table' AND name NOT LIKE 'sqlite_%'"
        ).fetchall()
    finally:
        con.close()
    return {r[0].lower() for r in rows}


def table_overlap(a: Path, b: Path) -> float:
    """Jaccard similarity of the two databases' lower-cased table names: a cheap fingerprint
    for the same data published under two database names."""
    ta, tb = _tables(a), _tables(b)
    return len(ta & tb) / len(ta | tb) if ta | tb else 0.0
