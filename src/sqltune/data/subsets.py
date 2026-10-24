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


def split_by_db(
    examples: list[Example], fraction: float, seed: int
) -> tuple[list[Example], list[Example]]:
    """(train, val): whole databases, in a seeded shuffled order, go to val until it holds at
    least `fraction` of the questions; at least one database always stays in train. Questions
    of one database are near-duplicates in style, so a per-question split would leak."""
    dbs = sorted({e.db_id for e in examples})
    if fraction <= 0 or len(dbs) < 2:
        return examples, []
    random.Random(seed).shuffle(dbs)
    sizes = {db: sum(1 for e in examples if e.db_id == db) for db in dbs}
    val_dbs: set[str] = set()
    for db in dbs[:-1]:
        if sum(sizes[d] for d in val_dbs) >= fraction * len(examples):
            break
        val_dbs.add(db)
    return (
        [e for e in examples if e.db_id not in val_dbs],
        [e for e in examples if e.db_id in val_dbs],
    )


def select_subset(examples: list[Example], cfg: DataConfig) -> list[Example]:
    if cfg.subset == "full":
        return examples
    shuffled = list(examples)
    random.Random(cfg.seed).shuffle(shuffled)
    return sorted(shuffled[: cfg.dev_small_size], key=lambda e: e.id)
