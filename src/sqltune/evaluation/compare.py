"""Decide whether two query results are the same answer (execution accuracy).

Purpose: Two differently written queries are equally correct when they return the same rows;
comparing results instead of SQL text gives credit for any valid formulation.
Flow: executor.py (gold rows, predicted rows) -> results_match() -> Outcome.
Params: order_matters (from gold_has_order_by), column_order_matters.
Rules: rows compared as a multiset (duplicates count) unless the gold query sorts at its
outermost level; columns compared in order (strict EX, stricter than the official Spider
evaluator) or, for the tolerant EX, under any column permutation that makes the results match
(what the official test-suite evaluator accepts, so ex_col_tolerant is the number comparable
to published Spider EX).
Try: `results_match([(1, "a")], [("a", 1)], False, column_order_matters=False)` -> True.
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Iterator
from typing import Any, Literal

Rows = list[tuple[Any, ...]]

# Column assignments tried before giving up on the tolerant match (reported as not equal).
MAX_PERMUTATIONS = 10_000

_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _same(gold: Rows, pred: Rows, order_matters: bool) -> bool:
    if order_matters:
        return gold == pred
    return Counter(gold) == Counter(pred)


def _column_permutations(gold: Rows, pred: Rows, order_matters: bool) -> Iterator[tuple[int, ...]]:
    """Yield permutations p with pred column p[j] holding the same values as gold column j.

    Backtracking with symmetry breaking: pred columns whose values are identical are
    interchangeable, so only the first unused one of each identical group is tried. This
    keeps wide results with many equal columns (all-NULL, all-0) from exploding."""
    ncols = len(gold[0])

    def key(rows: Rows, i: int) -> Any:
        values = [r[i] for r in rows]
        return tuple(values) if order_matters else frozenset(Counter(values).items())

    gold_keys = [key(gold, j) for j in range(ncols)]
    pred_cols = [tuple(r[i] for r in pred) for i in range(ncols)]
    candidates = [[i for i in range(ncols) if key(pred, i) == gold_keys[j]] for j in range(ncols)]
    tried = 0

    def extend(prefix: list[int], used: set[int]) -> Iterator[tuple[int, ...]]:
        nonlocal tried
        if len(prefix) == ncols:
            tried += 1
            yield tuple(prefix)
            return
        seen: set[tuple[Any, ...]] = set()
        for i in candidates[len(prefix)]:
            if i in used or pred_cols[i] in seen or tried >= MAX_PERMUTATIONS:
                continue
            seen.add(pred_cols[i])
            yield from extend([*prefix, i], used | {i})

    yield from extend([], set())


def results_match(
    gold: Rows,
    pred: Rows,
    order_matters: bool,
    column_order_matters: bool = True,
    row_compare: Literal["multiset", "set"] = "multiset",
) -> bool:
    if row_compare == "set":  # BIRD: set(gold) == set(pred), order and duplicates ignored
        gold, pred = list(dict.fromkeys(gold)), list(dict.fromkeys(pred))
        order_matters = False
    if _same(gold, pred, order_matters):
        return True
    if column_order_matters or not gold or not pred or len(gold) != len(pred):
        return False
    if len(gold[0]) != len(pred[0]):
        return False
    for perm in _column_permutations(gold, pred, order_matters):
        if _same(gold, [tuple(r[i] for i in perm) for r in pred], order_matters):
            return True
    return False


def gold_has_order_by(sql: str) -> bool:
    """True when ORDER BY appears outside string literals and outside parentheses."""
    masked: list[str] = []
    depth = 0
    quote: str | None = None
    for ch in sql:
        if quote:
            masked.append(" ")
            if ch == quote:
                quote = None
            continue
        if ch in ("'", '"'):
            quote = ch
            masked.append(" ")
        elif ch == "(":
            depth += 1
            masked.append(" ")
        elif ch == ")":
            depth -= 1
            masked.append(" ")
        else:
            masked.append(ch if depth == 0 else " ")
    return bool(_ORDER_BY.search("".join(masked)))
