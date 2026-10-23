"""Serialize a SQLite database's schema as the CREATE TABLE text the model sees.

Purpose: The model can only write correct SQL against tables and columns it is shown; the
original DDL keeps types, primary and foreign keys, which help join choice. Optional example
rows show real values and spellings (how dates, codes and names look), which targets the
"invented column / wrong value" errors.
Flow: Example.db_path -> schema_text() -> prompting/builder.py.
Params: PromptConfig.sample_rows (0 = DDL only).
Try: `print(schema_text(Path("data/spider/database/concert_singer/concert_singer.sqlite"), 3))`.
"""

from __future__ import annotations

import sqlite3
from functools import lru_cache
from pathlib import Path

# Longest cell value shown in example rows; longer text is cut and marked with "...".
MAX_VALUE_CHARS = 40


def _cell(value: object) -> str:
    text = repr(value) if isinstance(value, str) else str(value)
    return text if len(text) <= MAX_VALUE_CHARS else text[: MAX_VALUE_CHARS - 3] + "..."


def _example_rows(con: sqlite3.Connection, table: str, n: int) -> str:
    cur = con.execute(f'SELECT * FROM "{table}" LIMIT {int(n)}')
    columns = [d[0] for d in cur.description]
    rows = cur.fetchall()
    lines = [" | ".join(columns), *(" | ".join(_cell(v) for v in row) for row in rows)]
    return f"/* {len(rows)} example rows from {table}:\n" + "\n".join(lines) + "\n*/"


@lru_cache(maxsize=512)
def schema_text(db_path: Path, sample_rows: int = 0) -> str:
    """CREATE TABLE statements of every user table, sorted by name, each optionally followed by
    its first `sample_rows` rows."""
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.text_factory = lambda raw: raw.decode("utf-8", errors="replace")
    try:
        tables = con.execute(
            "SELECT name, sql FROM sqlite_master "
            "WHERE type = 'table' AND name NOT LIKE 'sqlite_%' AND sql IS NOT NULL "
            "ORDER BY name"
        ).fetchall()
        parts = []
        for name, sql in tables:
            part = sql.strip()
            if sample_rows > 0:
                part += "\n" + _example_rows(con, name, sample_rows)
            parts.append(part)
    finally:
        con.close()
    return "\n\n".join(parts)
