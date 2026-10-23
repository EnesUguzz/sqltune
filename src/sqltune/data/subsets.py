"""Pick the evaluation subset: the whole split, or a fixed random sample for smoke tests.

Purpose: A small, reproducible slice (dev_small) lets the pipeline be checked in seconds
without changing which questions a given seed selects between runs.
Flow: load_spider() -> select_subset() -> runner.
Params: DataConfig.subset, dev_small_size, seed.
Try: `select_subset(load_spider(cfg, "dev"), cfg.model_copy(update={"subset": "dev_small"}))`.
"""

from __future__ import annotations

import random

from sqltune.config.models import DataConfig
from sqltune.domain import Example


def select_subset(examples: list[Example], cfg: DataConfig) -> list[Example]:
    if cfg.subset == "full":
        return examples
    shuffled = list(examples)
    random.Random(cfg.seed).shuffle(shuffled)
    return sorted(shuffled[: cfg.dev_small_size], key=lambda e: e.id)
