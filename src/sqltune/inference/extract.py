"""Pull one SQL statement out of a model's free-text answer.

Purpose: Models wrap SQL in code fences, add explanations or chain several statements;
scoring must run exactly one query, the one the model meant as its answer.
Flow: Completion.text -> extract_sql() -> evaluation/executor.py (None = 'no_sql' outcome).
Params: none. Order: drop <think>...</think>; last ```sql block; else last ``` block;
else text from the first SELECT/WITH; then keep only the first complete statement.
Try: `extract_sql("Sure!\\n```sql\\nSELECT 1;\\n```")`.
"""

from __future__ import annotations

import re
import sqlite3

_THINK = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_SQL_FENCE = re.compile(r"```sql[ \t]*\n?(.*?)```", re.DOTALL | re.IGNORECASE)
_ANY_FENCE = re.compile(r"```[A-Za-z]*[ \t]*\n?(.*?)```", re.DOTALL)
_SQL_START = re.compile(r"\b(SELECT|WITH)\b", re.IGNORECASE)


def _first_statement(sql: str) -> str:
    """Text up to the first ';' that ends a complete statement (';' inside strings is kept)."""
    for i, ch in enumerate(sql):
        if ch == ";" and sqlite3.complete_statement(sql[: i + 1]):
            return sql[:i]
    return sql


def extract_sql(text: str) -> str | None:
    text = _THINK.sub("", text)
    blocks = _SQL_FENCE.findall(text) or _ANY_FENCE.findall(text)
    if blocks:
        candidate = blocks[-1]
    else:
        match = _SQL_START.search(text)
        if match is None:
            return None
        candidate = text[match.start() :]
    sql = _first_statement(candidate).strip().rstrip(";").strip()
    return sql or None
