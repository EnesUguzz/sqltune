"""Shared fixtures: a two-table SQLite DB and a mini Spider folder laid out like the real one."""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path

import pytest

from sqltune.config.models import DataConfig

SINGERS = [
    (1, "Ada", 25, "France"),
    (2, "Bo", 34, "Spain"),
    (3, "Cy", 41, "France"),
    (4, "Di", 29, "Italy"),
    (5, "Ed", 52, "Spain"),
    (6, "Fy", 37, "Italy"),
]
CONCERTS = [(1, 1, 2014), (2, 2, 2015), (3, 2, 2015), (4, 3, 2016), (5, 5, 2014), (6, 6, 2017)]


def make_tiny_db(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    con.execute(
        "CREATE TABLE singer (singer_id INTEGER PRIMARY KEY, name TEXT, age INTEGER, country TEXT)"
    )
    con.execute(
        "CREATE TABLE concert (concert_id INTEGER PRIMARY KEY, singer_id INTEGER "
        "REFERENCES singer(singer_id), year INTEGER)"
    )
    con.executemany("INSERT INTO singer VALUES (?, ?, ?, ?)", SINGERS)
    con.executemany("INSERT INTO concert VALUES (?, ?, ?)", CONCERTS)
    con.commit()
    con.close()
    return path


@pytest.fixture
def tiny_db(tmp_path: Path) -> Path:
    return make_tiny_db(tmp_path / "tiny.sqlite")


def _records(n: int) -> list[dict[str, str]]:
    return [
        {"db_id": "tiny", "question": f"Question {i}?", "query": "SELECT count(*) FROM singer"}
        for i in range(n)
    ]


@pytest.fixture
def tiny_spider(tmp_path: Path) -> DataConfig:
    """data_dir with spider/{dev.json, train_spider.json, database/tiny/tiny.sqlite}."""
    root = tmp_path / "data" / "spider"
    make_tiny_db(root / "database" / "tiny" / "tiny.sqlite")
    (root / "dev.json").write_text(json.dumps(_records(6)), encoding="utf-8")
    (root / "train_spider.json").write_text(json.dumps(_records(4)), encoding="utf-8")
    return DataConfig(data_dir=tmp_path / "data")
