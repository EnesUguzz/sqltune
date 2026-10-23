from __future__ import annotations

import pytest

from sqltune.training.train import check_loss_mask

HEADER = [7, 8, 9]  # stands in for "<|im_start|>assistant\n"


class FakeTokenizer:
    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        return HEADER


class Rows:
    def __init__(self, rows: list[list[int]]) -> None:
        self.rows = rows

    def tolist(self) -> list[list[int]]:
        return self.rows


IDS = [1, 2, 3, *HEADER, 40, 41, 42]  # prompt (6 tokens incl. header) + 3 completion tokens


def _batch(labels: list[int]) -> dict[str, Rows]:
    return {"input_ids": Rows([IDS]), "labels": Rows([labels])}


def test_correct_mask_passes() -> None:
    check_loss_mask(_batch([-100] * 6 + [40, 41, 42]), FakeTokenizer())


def test_unmasked_prompt_fails() -> None:
    with pytest.raises(AssertionError, match="prompt tokens"):
        check_loss_mask(_batch(IDS), FakeTokenizer())


def test_masked_completion_fails() -> None:
    with pytest.raises(AssertionError, match="first completion token"):
        check_loss_mask(_batch([-100] * 9), FakeTokenizer())


def test_missing_header_fails() -> None:
    batch = {"input_ids": Rows([[1, 2, 3]]), "labels": Rows([[-100, -100, 3]])}
    with pytest.raises(AssertionError, match="header"):
        check_loss_mask(batch, FakeTokenizer())
