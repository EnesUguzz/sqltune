"""Choose solved example questions to show the model before the real one (few-shot).

Purpose: The cheapest way to teach output format and SQL conventions without training; the
fair baseline a fine-tune has to beat. Shots come from Spider train and never from the
database being asked about, so no answer can leak.
Flow: target Example + Spider train pool -> pick_shots() -> build_messages(shots=...).
Params: PromptConfig.few_shot_k, PromptConfig.few_shot_seed.
Try: `pick_shots(load_spider(cfg, "dev")[0], load_spider(cfg, "train"), 3, 13)`.
"""

from __future__ import annotations

import random
import zlib

from sqltune.domain import Example


def pick_shots(target: Example, pool: list[Example], k: int, seed: int) -> list[Example]:
    """k pool examples from databases other than target.db_id, at most one per database;
    deterministic per (target, seed)."""
    rng = random.Random(seed * 1_000_003 + zlib.crc32(target.id.encode()))
    candidates = [ex for ex in pool if ex.db_id != target.db_id]
    rng.shuffle(candidates)
    shots: list[Example] = []
    used: set[str] = set()
    for ex in candidates:
        if ex.db_id not in used:
            shots.append(ex)
            used.add(ex.db_id)
        if len(shots) == k:
            break
    return shots
