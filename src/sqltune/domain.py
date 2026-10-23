"""Core records shared by data loading, prompting, inference and scoring.

Purpose: One typed shape for a text-to-SQL example, so every module agrees on field names.
Flow: data/spider.py -> Example -> prompting / runner / training.
Params: none.
Try: `load_spider(DataConfig(), "dev")[0]`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel

# Scoring result per example. gold_error = the reference SQL itself fails; such examples are
# left out of the EX denominator and counted separately.
Outcome = Literal[
    "correct", "wrong_result", "exec_error", "timeout", "too_many_rows", "no_sql", "gold_error"
]


class ExecResult(BaseModel):
    ok: bool
    rows: list[tuple[Any, ...]] | None = None
    error_type: Literal["exec_error", "timeout", "too_many_rows"] | None = None
    error: str | None = None
    elapsed_ms: float


class Example(BaseModel):
    id: str
    split: str
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
