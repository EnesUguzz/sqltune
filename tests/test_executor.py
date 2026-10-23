from __future__ import annotations

import hashlib
import time
from pathlib import Path

from sqltune.config.models import EvalConfig
from sqltune.domain import Example
from sqltune.evaluation.compare import gold_has_order_by, results_match
from sqltune.evaluation.executor import execute, score_example

CFG = EvalConfig()


def _ex(db: Path, gold: str) -> Example:
    return Example(id="t-0", split="dev", db_id="tiny", db_path=db, question="q", gold_sql=gold)


def test_equivalent_queries_correct(tiny_db: Path) -> None:
    ex = _ex(tiny_db, "SELECT name FROM singer WHERE age > 30")
    outcome, _, res = score_example(ex, "SELECT s.name FROM singer AS s WHERE NOT s.age <= 30", CFG)
    assert outcome == "correct"
    assert res is not None and res.ok


def test_order_ignored_without_order_by() -> None:
    assert results_match([(1,), (2,)], [(2,), (1,)], order_matters=False)


def test_order_respected_with_order_by(tiny_db: Path) -> None:
    ex = _ex(tiny_db, "SELECT name FROM singer ORDER BY age DESC")
    outcome, _, _ = score_example(ex, "SELECT name FROM singer ORDER BY age ASC", CFG)
    assert outcome == "wrong_result"
    outcome, _, _ = score_example(ex, "SELECT name FROM singer ORDER BY age DESC", CFG)
    assert outcome == "correct"


def test_duplicates_matter() -> None:
    assert not results_match([(1,), (1,)], [(1,)], order_matters=False)


def test_swapped_columns_strict_wrong_tolerant_correct(tiny_db: Path) -> None:
    ex = _ex(tiny_db, "SELECT name, age FROM singer")
    outcome, tolerant, _ = score_example(ex, "SELECT age, name FROM singer", CFG)
    assert outcome == "wrong_result"
    assert tolerant is True


def test_tolerant_needs_same_column_count() -> None:
    assert not results_match([(1, 2)], [(1, 2, 3)], order_matters=False, column_order_matters=False)


def test_correct_is_also_tolerant_correct(tiny_db: Path) -> None:
    ex = _ex(tiny_db, "SELECT name FROM singer")
    outcome, tolerant, _ = score_example(ex, "SELECT name FROM singer", CFG)
    assert (outcome, tolerant) == ("correct", True)


def test_tolerant_match_wide_identical_columns_is_fast() -> None:
    """Many columns with equal values must not explode into ncols^ncols candidates."""
    gold = [tuple([0] * 12)]
    pred = [tuple([0] * 11 + [1])]
    start = time.monotonic()
    assert not results_match(gold, pred, order_matters=False, column_order_matters=False)
    assert time.monotonic() - start < 1.0


def test_tolerant_match_finds_permutation_among_equal_columns() -> None:
    gold = [(1, 0, 0, 2), (3, 0, 0, 4)]
    pred = [(0, 2, 1, 0), (0, 4, 3, 0)]
    assert results_match(gold, pred, order_matters=True, column_order_matters=False)


def test_tolerant_match_int_equals_float() -> None:
    assert results_match([(1, "a")], [("a", 1.0)], order_matters=False, column_order_matters=False)


def test_gold_has_order_by() -> None:
    assert gold_has_order_by("SELECT a FROM t ORDER BY a")
    assert not gold_has_order_by("SELECT a FROM (SELECT a FROM t ORDER BY a LIMIT 3)")
    assert not gold_has_order_by("SELECT 'order by' FROM t")


def test_write_rejected_db_unchanged(tiny_db: Path) -> None:
    before = hashlib.sha256(tiny_db.read_bytes()).hexdigest()
    res = execute(tiny_db, "DELETE FROM singer", CFG)
    assert not res.ok
    assert res.error_type == "exec_error"
    assert hashlib.sha256(tiny_db.read_bytes()).hexdigest() == before


def test_timeout(tiny_db: Path) -> None:
    sql = "WITH RECURSIVE c(x) AS (SELECT 1 UNION ALL SELECT x + 1 FROM c) SELECT max(x) FROM c"
    start = time.monotonic()
    res = execute(tiny_db, sql, EvalConfig(exec_timeout_s=0.2))
    assert res.error_type == "timeout"
    assert time.monotonic() - start < 1.0


def test_too_many_rows(tiny_db: Path) -> None:
    res = execute(
        tiny_db, "SELECT * FROM singer, concert, singer AS s2", EvalConfig(max_result_rows=10)
    )
    assert res.error_type == "too_many_rows"


def test_gold_error(tiny_db: Path) -> None:
    outcome, _, _ = score_example(_ex(tiny_db, "SELECT nope FROM singer"), "SELECT 1", CFG)
    assert outcome == "gold_error"


def test_no_sql(tiny_db: Path) -> None:
    outcome, tolerant, res = score_example(_ex(tiny_db, "SELECT 1"), None, CFG)
    assert (outcome, tolerant, res) == ("no_sql", False, None)


def test_pred_exec_error(tiny_db: Path) -> None:
    outcome, _, res = score_example(_ex(tiny_db, "SELECT 1"), "SELECT nope FROM singer", CFG)
    assert outcome == "exec_error"
    assert res is not None and res.error
