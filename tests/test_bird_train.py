from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_tiny_db

from sqltune.config.models import DataConfig
from sqltune.data.bird_train import GOLD_FAILURES_FILE, load_bird_train, table_overlap


def make_bird_train(data_dir: Path, n_dbs: int = 4, per_db: int = 3) -> DataConfig:
    """data_dir with bird_train/{train.json, train_databases/<db>/<db>.sqlite}."""
    root = data_dir / "bird_train"
    records = []
    for d in range(n_dbs):
        db = f"bt{d}"
        make_tiny_db(root / "train_databases" / db / f"{db}.sqlite")
        for q in range(per_db):
            records.append(
                {
                    "db_id": db,
                    "question": f"Q{q} on {db}?",
                    "evidence": "" if q == 0 else f"hint {q}",
                    "SQL": "SELECT count(*) FROM singer",
                }
            )
    (root / "train.json").write_text(json.dumps(records), encoding="utf-8")
    (root / GOLD_FAILURES_FILE).write_text("[]", encoding="utf-8")  # as after dataset-review
    return DataConfig(data_dir=data_dir)


def test_bird_train_loader_fields(tmp_path: Path) -> None:
    examples = load_bird_train(make_bird_train(tmp_path / "data"))
    assert len(examples) == 12
    first = examples[0]
    assert first.id == "birdtrain-00000"
    assert first.split == "train"
    assert first.db_path.exists()
    assert first.evidence is None
    assert examples[1].evidence == "hint 1"


def test_failing_gold_ids_are_skipped(tmp_path: Path) -> None:
    cfg = make_bird_train(tmp_path / "data")
    path = cfg.data_dir / "bird_train" / GOLD_FAILURES_FILE
    path.write_text(json.dumps(["birdtrain-00001"]), encoding="utf-8")
    ids = [e.id for e in load_bird_train(cfg)]
    assert "birdtrain-00001" not in ids
    assert len(ids) == 11


def test_missing_gold_failures_file_is_an_error(tmp_path: Path) -> None:
    cfg = make_bird_train(tmp_path / "data")
    (cfg.data_dir / "bird_train" / GOLD_FAILURES_FILE).unlink()
    with pytest.raises(FileNotFoundError, match="dataset-review"):
        load_bird_train(cfg)


def test_table_overlap_flags_same_tables(tmp_path: Path) -> None:
    a = make_tiny_db(tmp_path / "a.sqlite")
    b = make_tiny_db(tmp_path / "b.sqlite")
    assert table_overlap(a, b) == 1.0


@pytest.mark.data
def test_bird_train_dbs_disjoint_from_eval() -> None:
    from sqltune.data.bird import load_bird
    from sqltune.data.spider import load_spider

    cfg = DataConfig()
    everything = cfg.model_copy(
        update={"bird_train_skip_failing_gold": False, "bird_train_exclude_dbs": []}
    )
    assert len(load_bird_train(everything)) == 9428
    train = load_bird_train(cfg.model_copy(update={"bird_train_skip_failing_gold": False}))
    eval_dbs = {e.db_id: e.db_path for e in load_bird(cfg.model_copy(update={"bird_split": "all"}))}
    eval_dbs |= {e.db_id: e.db_path for e in load_spider(cfg, "dev")}
    eval_dbs |= {e.db_id: e.db_path for e in load_spider(cfg, "test")}
    train_dbs = {e.db_id: e.db_path for e in train}
    assert not set(train_dbs) & set(eval_dbs)
    flagged = [
        (t, e)
        for t, tp in train_dbs.items()
        for e, ep in eval_dbs.items()
        if table_overlap(tp, ep) >= 0.5
    ]
    # found on 2026-10-24: world == Spider world_1 (same data), airline ~ flight_2 (same
    # table names); both are in DataConfig.bird_train_exclude_dbs, so nothing is left here
    assert flagged == [], flagged


def test_excluded_dbs_are_not_loaded(tmp_path: Path) -> None:
    cfg = make_bird_train(tmp_path / "data").model_copy(update={"bird_train_exclude_dbs": ["bt1"]})
    assert {e.db_id for e in load_bird_train(cfg)} == {"bt0", "bt2", "bt3"}


@pytest.mark.data
def test_m1b_training_dbs_disjoint_from_bird_eval() -> None:
    """Every database the M1b run trains on vs BIRD Mini-Dev (name or table similarity >= 0.5),
    and BIRD train vs Spider dev/test. Spider train vs Spider dev/test is the official Spider
    split, which has a few look-alike generic schemas (e.g. tracking_orders ~ e_commerce);
    it is kept as everyone trains on it."""
    from sqltune.config.loader import load_config
    from sqltune.data.bird import load_bird
    from sqltune.data.spider import load_spider

    cfg = load_config(Path("experiments/configs/m1b/train/mix.yaml")).data
    spider_train = {e.db_id: e.db_path for e in load_spider(cfg, "train")}
    bird_train = {e.db_id: e.db_path for e in load_bird_train(cfg)}
    minidev = {e.db_id: e.db_path for e in load_bird(cfg.model_copy(update={"bird_split": "all"}))}
    spider_eval = {e.db_id: e.db_path for s in ("dev", "test") for e in load_spider(cfg, s)}  # type: ignore[arg-type]
    pairs = [(t, tp, minidev) for t, tp in (spider_train | bird_train).items()]
    pairs += [(t, tp, spider_eval) for t, tp in bird_train.items()]
    flagged = [
        (t, v)
        for t, tp, evals in pairs
        for v, vp in evals.items()
        if t == v or table_overlap(tp, vp) >= 0.5
    ]
    # found on 2026-10-24: Spider train formula_1 == Mini-Dev formula_1 (diag) and soccer_1 ~
    # european_football_2 (holdout, 0.86); excluded in m1b mix.yaml (spider_train_exclude_dbs)
    assert flagged == [], flagged
