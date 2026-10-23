from __future__ import annotations

from pathlib import Path

import pytest

from sqltune.config.models import DataConfig
from sqltune.data import download
from sqltune.data.spider import load_spider
from sqltune.data.subsets import select_subset
from sqltune.domain import Example


def test_load_dev_builds_examples(tiny_spider: DataConfig) -> None:
    examples = load_spider(tiny_spider, "dev")
    assert len(examples) == 6
    first = examples[0]
    assert first.id == "spider-dev-0000"
    assert first.split == "dev"
    assert first.db_id == "tiny"
    assert first.db_path.exists()
    assert first.gold_sql == "SELECT count(*) FROM singer"


def test_load_train_ids(tiny_spider: DataConfig) -> None:
    ids = [e.id for e in load_spider(tiny_spider, "train")]
    assert ids == [f"spider-train-{i:04d}" for i in range(4)]


def _examples(n: int) -> list[Example]:
    return [
        Example(
            id=f"spider-dev-{i:04d}",
            split="dev",
            db_id="tiny",
            db_path=Path("x.sqlite"),
            question="q",
            gold_sql="SELECT 1",
        )
        for i in range(n)
    ]


def test_dev_small_deterministic() -> None:
    exs = _examples(50)
    cfg = DataConfig(subset="dev_small", dev_small_size=10, seed=13)
    a = [e.id for e in select_subset(exs, cfg)]
    b = [e.id for e in select_subset(exs, cfg)]
    c = [e.id for e in select_subset(exs, cfg.model_copy(update={"seed": 14}))]
    assert a == b
    assert a != c
    assert len(a) == 10
    assert a == sorted(a)
    assert len(select_subset(_examples(3), cfg)) == 3


def test_full_subset_returns_all() -> None:
    exs = _examples(7)
    assert select_subset(exs, DataConfig(subset="full")) == exs


def test_download_idempotent(tiny_spider: DataConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise AssertionError("downloader must not be called when data exists")

    monkeypatch.setattr(download, "_fetch_zip", fail)
    assert download.download_spider(tiny_spider) == tiny_spider.data_dir / "spider"


@pytest.mark.data
def test_real_spider_counts() -> None:
    cfg = DataConfig()
    assert len(load_spider(cfg, "dev")) == 1034
    assert len(load_spider(cfg, "train")) == 7000


def test_include_train_others(tiny_spider: DataConfig) -> None:
    import json as _json

    root = tiny_spider.data_dir / "spider"
    rec = [{"db_id": "tiny", "question": "Other?", "query": "SELECT 1"}]
    (root / "train_others.json").write_text(_json.dumps(rec), encoding="utf-8")
    cfg = tiny_spider.model_copy(update={"include_train_others": True})
    ids = [e.id for e in load_spider(cfg, "train")]
    assert len(ids) == 5
    assert ids[-1] == "spider-trainothers-0000"
    assert len(load_spider(tiny_spider, "train")) == 4  # off by default


def test_train_fraction_deterministic() -> None:
    from sqltune.data.subsets import select_fraction

    exs = _examples(100)
    a = select_fraction(exs, 0.25, seed=13)
    assert len(a) == 25
    assert a == select_fraction(exs, 0.25, seed=13)
    assert select_fraction(exs, 1.0, seed=13) == exs
    assert set(e.id for e in select_fraction(exs, 0.125, seed=13)) <= set(
        e.id for e in select_fraction(exs, 0.25, seed=13)
    )  # nested: a smaller fraction is a subset of a larger one
