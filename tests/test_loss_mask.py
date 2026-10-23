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


def test_packed_row_checked_per_sequence() -> None:
    """Padding-free batches concatenate examples in one row; position_ids restart at 0."""
    good = [-100] * 6 + [40, 41, 42]
    batch = {
        "input_ids": Rows([IDS + IDS]),
        "labels": Rows([good + good]),
        "position_ids": Rows([list(range(9)) + list(range(9))]),
    }
    check_loss_mask(batch, FakeTokenizer())
    batch["labels"] = Rows([good + IDS])  # second sequence trains on its prompt
    with pytest.raises(AssertionError, match="prompt tokens"):
        check_loss_mask(batch, FakeTokenizer())


def test_completion_must_end_with_eos() -> None:
    """The last trained token must be <|im_end|>, or the model never learns to stop."""
    labels = [-100] * 6 + [40, 41, 42]
    with pytest.raises(AssertionError, match="end"):
        check_loss_mask(_batch(labels), FakeTokenizer(), eos_id=99)
    ids_eos = [1, 2, 3, *HEADER, 40, 41, 99]
    batch = {"input_ids": Rows([ids_eos]), "labels": Rows([[-100] * 6 + [40, 41, 99]])}
    check_loss_mask(batch, FakeTokenizer(), eos_id=99)


def test_missing_header_fails() -> None:
    batch = {"input_ids": Rows([[1, 2, 3]]), "labels": Rows([[-100, -100, 3]])}
    with pytest.raises(AssertionError, match="header"):
        check_loss_mask(batch, FakeTokenizer())


def test_warmup_steps_from_ratio() -> None:
    from sqltune.config.models import TrainConfig
    from sqltune.training.train import warmup_steps

    cfg = TrainConfig(
        warmup_ratio=0.03,
        num_train_epochs=2,
        per_device_train_batch_size=8,
        gradient_accumulation_steps=2,
    )
    assert warmup_steps(6918, cfg) == 26  # ceil(0.03 * ceil(6918 / 16) * 2) = ceil(25.95)
    assert warmup_steps(6918, cfg.model_copy(update={"max_steps": 50})) == 2
