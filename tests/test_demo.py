from __future__ import annotations

from pathlib import Path

from conftest import make_tiny_db

from sqltune.config.models import AppConfig
from sqltune.demo import answer, list_databases
from sqltune.inference.openai_compat import Completion


class ScriptedBackend:
    """Returns the given answers in order and records every request."""

    def __init__(self, *texts: str) -> None:
        self.texts = list(texts)
        self.requests: list[list[dict[str, str]]] = []

    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        self.requests += requests
        return [Completion(text=self.texts.pop(0), latency_ms=1.0) for _ in requests]


def test_answer_runs_sql_and_returns_columns(tmp_path: Path) -> None:
    db = make_tiny_db(tmp_path / "concert.sqlite")
    backend = ScriptedBackend("```sql\nSELECT count(*) AS n FROM singer\n```")
    out = answer(
        "How many singers?", db, "m1b_mix", AppConfig(run={"run_name": "demo"}), backend=backend
    )
    assert out["sql"] == "SELECT count(*) AS n FROM singer"
    assert out["columns"] == ["n"]
    assert out["n_rows"] == 1 and out["error"] is None and not out["retried"]
    user = backend.requests[0][-1]["content"]
    assert "CREATE TABLE" in user and "How many singers?" in user  # schema read from the file


def test_answer_retries_once_after_exec_error(tmp_path: Path) -> None:
    db = make_tiny_db(tmp_path / "concert.sqlite")
    backend = ScriptedBackend(
        "```sql\nSELECT nope FROM singer\n```", "```sql\nSELECT count(*) FROM singer\n```"
    )
    out = answer(
        "How many singers?", db, "m1b_mix", AppConfig(run={"run_name": "demo"}), backend=backend
    )
    assert out["retried"] and out["error"] is None
    assert "no such column" in backend.requests[1][-1]["content"]


def test_list_databases_finds_eval_dbs(tmp_path: Path) -> None:
    make_tiny_db(tmp_path / "spider" / "database" / "a" / "a.sqlite")
    make_tiny_db(tmp_path / "bird" / "dev_databases" / "b" / "b.sqlite")
    assert set(list_databases(tmp_path)) == {"spider/a", "bird/b"}
