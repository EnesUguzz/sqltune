from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_tiny_db

from sqltune.config.models import DataConfig
from sqltune.data.bird import load_bird, split_bird


def _bird_dir(tmp_path: Path, n_dbs: int = 6, per_db: int = 5) -> DataConfig:
    root = tmp_path / "data" / "bird"
    records = []
    for d in range(n_dbs):
        db = f"db{d}"
        make_tiny_db(root / "dev_databases" / db / f"{db}.sqlite")
        for q in range(per_db):
            records.append(
                {
                    "question_id": len(records),
                    "db_id": db,
                    "question": f"Q{q} on {db}?",
                    "evidence": "" if q == 0 else f"hint {q}",
                    "SQL": "SELECT count(*) FROM singer",
                    "difficulty": "simple",
                }
            )
    (root / "mini_dev_sqlite.json").write_text(json.dumps(records), encoding="utf-8")
    return DataConfig(data_dir=tmp_path / "data", dataset="bird", bird_split="all")


def test_bird_loader_fields(tmp_path: Path) -> None:
    examples = load_bird(_bird_dir(tmp_path))
    assert len(examples) == 30
    first = examples[0]
    assert first.id == "bird-0000"
    assert first.db_path.exists()
    assert first.evidence is None  # empty evidence string becomes None
    assert examples[1].evidence == "hint 1"
    assert first.difficulty == "simple"


def test_split_disjoint_dbs(tmp_path: Path) -> None:
    diag, holdout = split_bird(load_bird(_bird_dir(tmp_path)), seed=13)
    assert {e.db_id for e in diag}.isdisjoint({e.db_id for e in holdout})
    assert len(diag) + len(holdout) == 30
    assert 10 <= len(diag) <= 20


def test_split_deterministic(tmp_path: Path) -> None:
    examples = load_bird(_bird_dir(tmp_path))
    assert split_bird(examples, 13) == split_bird(examples, 13)


@pytest.mark.parametrize("split", ["diag", "holdout"])
def test_bird_split_selected_by_config(tmp_path: Path, split: str) -> None:
    cfg = _bird_dir(tmp_path).model_copy(update={"bird_split": split})
    diag, holdout = split_bird(load_bird(cfg.model_copy(update={"bird_split": "all"})), cfg.seed)
    expected = diag if split == "diag" else holdout
    assert load_bird(cfg) == expected


@pytest.mark.data
def test_real_bird_counts() -> None:
    cfg = DataConfig(dataset="bird", bird_split="all")
    examples = load_bird(cfg)
    assert len(examples) == 500
    diag, holdout = split_bird(examples, cfg.seed)
    print(f"diag {len(diag)} on {sorted({e.db_id for e in diag})}")
    print(f"holdout {len(holdout)} on {sorted({e.db_id for e in holdout})}")
