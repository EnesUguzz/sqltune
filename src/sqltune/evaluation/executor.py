"""Run SQL safely against a Spider SQLite file and score one prediction.

Purpose: Model SQL is untrusted: it may write, loop forever or return millions of rows.
Connections are read-only, every query has a deadline and a row cap, so one bad answer
cannot corrupt a database or stall a 1000-question run.
Flow: pred SQL + Example -> execute() (gold result cached) -> compare.results_match() -> Outcome.
Params: EvalConfig.exec_timeout_s, EvalConfig.gold_timeout_s, EvalConfig.max_result_rows.
Try: `execute(Path(".../concert_singer.sqlite"), "SELECT count(*) FROM singer", EvalConfig())`.
"""

from __future__ import annotations

import sqlite3
import threading
import time
from pathlib import Path

from sqltune.config.models import EvalConfig
from sqltune.domain import Example, ExecResult, Outcome
from sqltune.evaluation.compare import gold_has_order_by, results_match

# SQLite calls the progress handler every N virtual-machine instructions.
PROGRESS_EVERY = 10_000

_gold_cache: dict[tuple[Path, str, float, int], ExecResult] = {}
_gold_lock = threading.Lock()


def _decode(raw: bytes) -> str:
    return raw.decode("utf-8", errors="replace")  # a few Spider DBs hold non-UTF-8 text


def execute(db_path: Path, sql: str, cfg: EvalConfig) -> ExecResult:
    start = time.monotonic()
    deadline = start + cfg.exec_timeout_s

    def elapsed() -> float:
        return (time.monotonic() - start) * 1000

    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True, check_same_thread=False)
    con.text_factory = _decode
    con.set_progress_handler(lambda: int(time.monotonic() > deadline), PROGRESS_EVERY)
    try:
        cur = con.execute(sql)
        rows: list[tuple[object, ...]] = []
        while batch := cur.fetchmany(1000):
            rows.extend(batch)
            if len(rows) > cfg.max_result_rows:
                return ExecResult(
                    ok=False,
                    error_type="too_many_rows",
                    error=f"more than {cfg.max_result_rows} rows",
                    elapsed_ms=elapsed(),
                )
        columns = [d[0] for d in cur.description or []]
        return ExecResult(
            ok=True, rows=[tuple(r) for r in rows], columns=columns, elapsed_ms=elapsed()
        )
    except sqlite3.Error as exc:
        timed_out = time.monotonic() > deadline and "interrupt" in str(exc).lower()
        return ExecResult(
            ok=False,
            error_type="timeout" if timed_out else "exec_error",
            error=str(exc),
            elapsed_ms=elapsed(),
        )
    except Exception as exc:  # e.g. sqlite3.Warning for multiple statements, OverflowError
        return ExecResult(ok=False, error_type="exec_error", error=repr(exc), elapsed_ms=elapsed())
    finally:
        con.close()


def _gold_result(ex: Example, cfg: EvalConfig) -> ExecResult:
    cfg = cfg.model_copy(update={"exec_timeout_s": max(cfg.exec_timeout_s, cfg.gold_timeout_s)})
    key = (ex.db_path, ex.gold_sql, cfg.exec_timeout_s, cfg.max_result_rows)
    with _gold_lock:
        cached = _gold_cache.get(key)
    if cached is None:
        cached = execute(ex.db_path, ex.gold_sql, cfg)
        with _gold_lock:
            _gold_cache[key] = cached
    return cached


def score_example(
    ex: Example, pred_sql: str | None, cfg: EvalConfig
) -> tuple[Outcome, bool, ExecResult | None]:
    """(strict outcome, correct under column permutation, predicted execution result)."""
    gold = _gold_result(ex, cfg)
    if not gold.ok or gold.rows is None:
        return "gold_error", False, None
    if pred_sql is None:
        return "no_sql", False, None
    pred = execute(ex.db_path, pred_sql, cfg)
    if not pred.ok or pred.rows is None:
        assert pred.error_type is not None
        return pred.error_type, False, pred
    order = gold_has_order_by(ex.gold_sql)
    mode = cfg.row_compare
    if results_match(gold.rows, pred.rows, order, row_compare=mode):
        return "correct", True, pred
    tolerant = results_match(
        gold.rows, pred.rows, order, column_order_matters=False, row_compare=mode
    )
    return "wrong_result", tolerant, pred
