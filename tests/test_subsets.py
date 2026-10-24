from __future__ import annotations

from pathlib import Path

from sqltune.data.subsets import split_by_db
from sqltune.domain import Example


def _examples(sizes: dict[str, int]) -> list[Example]:
    return [
        Example(
            id=f"{db}-{i}",
            split="train",
            db_id=db,
            db_path=Path(f"/x/{db}.sqlite"),
            question="q",
            gold_sql="SELECT 1",
        )
        for db, n in sizes.items()
        for i in range(n)
    ]


SIZES = {f"db{i}": 10 + i for i in range(20)}  # 390 questions, largest DB 29


def test_split_by_db_disjoint_and_deterministic() -> None:
    examples = _examples(SIZES)
    train, val = split_by_db(examples, 0.05, seed=13)
    assert {e.db_id for e in train}.isdisjoint({e.db_id for e in val})
    assert len(train) + len(val) == len(examples)
    assert 0.05 <= len(val) / len(examples) < 0.05 + 29 / len(examples)
    assert split_by_db(examples, 0.05, seed=13) == (train, val)
    assert split_by_db(examples, 0.05, seed=14) != (train, val)


def test_split_by_db_zero_fraction() -> None:
    examples = _examples(SIZES)
    assert split_by_db(examples, 0.0, seed=13) == (examples, [])


def test_split_by_db_keeps_one_db_for_training() -> None:
    examples = _examples({"only": 5})
    assert split_by_db(examples, 0.5, seed=13) == (examples, [])
