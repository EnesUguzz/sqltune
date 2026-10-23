"""Read Spider questions into Example records.

Purpose: Turns the dataset's JSON files into typed examples that point at their SQLite file.
Flow: <data_dir>/spider/{train_spider,dev}.json -> load_spider() -> list[Example].
Params: DataConfig.data_dir; split "train" (7000) or "dev" (1034).
Try: `len(load_spider(DataConfig(), "dev"))`.
"""

from __future__ import annotations

import json
from typing import Literal

from sqltune.config.models import DataConfig
from sqltune.domain import Example

SPLIT_FILES = {"train": "train_spider.json", "dev": "dev.json"}


def load_spider(cfg: DataConfig, split: Literal["train", "dev"]) -> list[Example]:
    root = cfg.data_dir / "spider"
    records = json.loads((root / SPLIT_FILES[split]).read_text(encoding="utf-8"))
    return [
        Example(
            id=f"spider-{split}-{i:04d}",
            split=split,
            db_id=r["db_id"],
            db_path=root / "database" / r["db_id"] / f"{r['db_id']}.sqlite",
            question=r["question"],
            gold_sql=r["query"],
        )
        for i, r in enumerate(records)
    ]
