"""Read BIRD Mini-Dev (SQLite) into Example records and split it by database.

Purpose: BIRD is the generalization check: larger, messier databases than Spider, written
in a different style, with an "evidence" hint per question (BIRD's standard setup, included
in the prompt). The 500 questions are split by database into a diagnostic half (used for
decisions) and a holdout half (headline numbers, run once per final model).
Flow: <data_dir>/bird/{mini_dev_sqlite.json, dev_databases/<db>/<db>.sqlite} -> load_bird().
Params: DataConfig.bird_split ("diag" / "holdout" / "all"), DataConfig.seed.
Try: `len(load_bird(DataConfig(dataset="bird", bird_split="all")))` -> 500.
"""

from __future__ import annotations

import json
import random

from sqltune.config.models import DataConfig
from sqltune.domain import Example

QUESTIONS_FILE = "mini_dev_sqlite.json"


def _load_all(cfg: DataConfig) -> list[Example]:
    root = cfg.data_dir / "bird"
    records = json.loads((root / QUESTIONS_FILE).read_text(encoding="utf-8"))
    return [
        Example(
            id=f"bird-{i:04d}",
            split="minidev",
            db_id=r["db_id"],
            db_path=root / "dev_databases" / r["db_id"] / f"{r['db_id']}.sqlite",
            question=r["question"],
            gold_sql=r["SQL"],
            evidence=(r.get("evidence") or "").strip() or None,
            difficulty=r.get("difficulty"),
        )
        for i, r in enumerate(records)
    ]


def split_bird(examples: list[Example], seed: int) -> tuple[list[Example], list[Example]]:
    """(diag, holdout): the subset of databases whose question count is closest to half goes
    to diag (ties broken by a `seed`-shuffled order); the rest is holdout. No database is in
    both. Mini-Dev has 11 databases, so all 2^11 subsets are checked."""
    dbs = sorted({e.db_id for e in examples})
    random.Random(seed).shuffle(dbs)
    sizes = [sum(1 for e in examples if e.db_id == db) for db in dbs]
    half = len(examples) / 2
    best = min(
        range(1, 2 ** len(dbs) - 1),
        key=lambda mask: abs(sum(s for i, s in enumerate(sizes) if mask >> i & 1) - half),
    )
    diag_dbs = {db for i, db in enumerate(dbs) if best >> i & 1}
    diag = [e for e in examples if e.db_id in diag_dbs]
    holdout = [e for e in examples if e.db_id not in diag_dbs]
    return diag, holdout


def load_bird(cfg: DataConfig) -> list[Example]:
    examples = _load_all(cfg)
    if cfg.bird_split == "all":
        return examples
    diag, holdout = split_bird(examples, cfg.seed)
    return diag if cfg.bird_split == "diag" else holdout
