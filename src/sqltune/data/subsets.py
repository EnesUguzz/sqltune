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


def select_fraction(examples: list[Example], fraction: float, seed: int) -> list[Example]:
    """First round(fraction * n) examples of one fixed shuffle, in original order; fractions
    are nested (0.125 is a subset of 0.25 for the same seed), so a learning curve only adds
    data from point to point."""
    if fraction >= 1.0:
        return examples
    order = list(range(len(examples)))
    random.Random(seed).shuffle(order)
    keep = sorted(order[: round(fraction * len(examples))])
    return [examples[i] for i in keep]


def select_subset(examples: list[Example], cfg: DataConfig) -> list[Example]:
    if cfg.subset == "full":
        return examples
    shuffled = list(examples)
    random.Random(cfg.seed).shuffle(shuffled)
    return sorted(shuffled[: cfg.dev_small_size], key=lambda e: e.id)
