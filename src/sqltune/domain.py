"""Core records shared by data loading, prompting, inference and scoring.

Purpose: One typed shape for a text-to-SQL example, so every module agrees on field names.
Flow: data/spider.py -> Example -> prompting / runner / training.
Params: none.
Try: `load_spider(DataConfig(), "dev")[0]`.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel


class Example(BaseModel):
    id: str
    split: str
    db_id: str
    db_path: Path
    question: str
    gold_sql: str
