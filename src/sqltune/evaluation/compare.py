"""Decide whether two query results are the same answer (execution accuracy).

Purpose: Two differently written queries are equally correct when they return the same rows;
comparing results instead of SQL text gives credit for any valid formulation.
Flow: executor.py (gold rows, predicted rows) -> results_match() -> Outcome.
Params: order_matters (from gold_has_order_by), column_order_matters.
Rules: rows compared as a multiset (duplicates count) unless the gold query sorts at its
outermost level; columns compared in order (strict EX, Spider's convention) or, for the
tolerant EX, under any column permutation that makes the results match.
Try: `results_match([(1, "a")], [("a", 1)], False, column_order_matters=False)` -> True.
"""

from __future__ import annotations

import itertools
import re
from collections import Counter
from typing import Any

Rows = list[tuple[Any, ...]]

_ORDER_BY = re.compile(r"\bORDER\s+BY\b", re.IGNORECASE)


def _same(gold: Rows, pred: Rows, order_matters: bool) -> bool:
    if order_matters:
        return gold == pred
    return Counter(gold) == Counter(pred)


def _column_permutations(gold: Rows, pred: Rows, order_matters: bool) -> list[tuple[int, ...]]:
    """Permutations p with pred column p[j] holding the same values as gold column j."""
    ncols = len(gold[0])

    def col(rows: Rows, i: int) -> list[Any]:
        return [r[i] for r in rows]

    candidates = []
    for j in range(ncols):
        g = col(gold, j)
        if order_matters:
            options = [i for i in range(ncols) if col(pred, i) == g]
        else:
            options = [i for i in range(ncols) if Counter(col(pred, i)) == Counter(g)]
        candidates.append(options)
    return [p for p in itertools.product(*candidates) if len(set(p)) == ncols]


def results_match(
    gold: Rows, pred: Rows, order_matters: bool, column_order_matters: bool = True
) -> bool:
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
