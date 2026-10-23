"""Serialize a SQLite database's schema as the CREATE TABLE text the model sees.

Purpose: The model can only write correct SQL against tables and columns it is shown; the
original DDL keeps types, primary and foreign keys, which help join choice.
Flow: Example.db_path -> schema_text() -> prompting/builder.py.
Params: none (example rows per table come in M1).
Try: `print(schema_text(Path("data/spider/database/concert_singer/concert_singer.sqlite")))`.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=512)
def schema_text(db_path: Path) -> str:
    """Original CREATE TABLE statements of every user table, sorted by table name."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        rows = con.execute(
            "SELECT sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
    finally:
        con.close()
    return "\n\n".join(sql.strip() for (sql,) in rows)
