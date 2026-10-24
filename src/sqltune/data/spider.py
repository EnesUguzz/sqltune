"""Read Spider questions into Example records.

Purpose: Turns the dataset's JSON files into typed examples that point at their SQLite file.
Flow: <data_dir>/spider/{train_spider,dev,test}.json -> load_spider() -> list[Example].
Params: DataConfig.data_dir; split "train" (7000), "dev" (1034) or "test" (2147, its own
test_database/ folder; run once per final model).
Try: `len(load_spider(DataConfig(), "dev"))`.
"""

from __future__ import annotations

import json
from typing import Literal

from sqltune.config.models import DataConfig
from sqltune.domain import Example

SPLIT_FILES = {"train": "train_spider.json", "dev": "dev.json", "test": "test.json"}
DB_FOLDERS = {"train": "database", "dev": "database", "test": "test_database"}


def _read(cfg: DataConfig, filename: str, split: str, prefix: str) -> list[Example]:
    root = cfg.data_dir / "spider"
    records = json.loads((root / filename).read_text(encoding="utf-8"))
    return [
        Example(
            id=f"{prefix}-{i:04d}",
            split=split,
            db_id=r["db_id"],
            db_path=root / DB_FOLDERS[split] / r["db_id"] / f"{r['db_id']}.sqlite",
            question=r["question"],
            gold_sql=r["query"],
        )
        for i, r in enumerate(records)
    ]


def load_spider(cfg: DataConfig, split: Literal["train", "dev", "test"]) -> list[Example]:
    examples = _read(cfg, SPLIT_FILES[split], split, f"spider-{split}")
    if split == "train" and cfg.include_train_others:
        examples += _read(cfg, "train_others.json", "train", "spider-trainothers")
    if split == "train" and cfg.spider_train_exclude_dbs:
        excluded = set(cfg.spider_train_exclude_dbs)
        examples = [e for e in examples if e.db_id not in excluded]
    return examples
