from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from sqltune.config.models import AppConfig, DataConfig, PromptConfig
from sqltune.domain import Example
from sqltune.prompting.builder import build_messages
from sqltune.training.sft_data import build_sft_dataset, to_sft_record

REPO = Path(__file__).parents[1]
PROMPT_CFG = PromptConfig(
    system_template=REPO / "prompts/sql_system.md",
    single_template=REPO / "prompts/sql_single.md",
)


def _example(db: Path) -> Example:
    return Example(
        id="spider-train-0000",
        split="train",
        db_id="tiny",
        db_path=db,
        question="How many singers?",
        gold_sql="  SELECT count(*) FROM singer  ",
    )


def test_prompt_identical_to_eval(tiny_db: Path) -> None:
    ex = _example(tiny_db)
    assert to_sft_record(ex, PROMPT_CFG)["prompt"] == build_messages(ex, PROMPT_CFG)


def test_completion_is_fenced_gold(tiny_db: Path) -> None:
    completion = to_sft_record(_example(tiny_db), PROMPT_CFG)["completion"]
    assert completion == [
        {"role": "assistant", "content": "```sql\nSELECT count(*) FROM singer\n```"}
    ]


class CharTokenizer:
    """One token per character of the joined message contents."""

    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> list[int]:
        return [0] * sum(len(m["content"]) for m in messages)


def test_overlong_records_dropped_and_counted(tiny_spider: DataConfig) -> None:
    pytest.importorskip("datasets", reason="needs the train extra: uv sync --extra train")
    cfg = AppConfig(data=tiny_spider, prompt=PROMPT_CFG, run={"run_name": "t"})
    _, dropped = build_sft_dataset(cfg, CharTokenizer())
    assert dropped == 0
    short = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"max_seq_length": 256})}
    )
    dataset, dropped = build_sft_dataset(short, CharTokenizer())
    assert dropped == 4  # every tiny record is > 256 characters once the schema is included
    assert len(dataset) == 0


@pytest.mark.data
def test_prompt_tokens_match_generation_prompt() -> None:
    """The tokens TRL treats as prompt (loss-masked) are exactly what vLLM will be given.

    TRL tokenizes prompt (add_generation_prompt=True) and prompt+completion separately and
    masks the common prefix; that prefix must be the whole generation prompt."""
    from transformers import AutoTokenizer

    from sqltune.data.spider import load_spider

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    ex = load_spider(DataConfig(), "train")[0]
    record = to_sft_record(ex, PromptConfig())
    prompt_ids = tok.apply_chat_template(
        record["prompt"], add_generation_prompt=True, tokenize=True
    )
    full_ids = tok.apply_chat_template(record["prompt"] + record["completion"], tokenize=True)
    if hasattr(prompt_ids, "keys"):  # transformers 5 returns a BatchEncoding
        prompt_ids, full_ids = prompt_ids["input_ids"], full_ids["input_ids"]
    assert full_ids[: len(prompt_ids)] == prompt_ids
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    assert im_end in full_ids[len(prompt_ids) :]
    newline = tok.encode("\n", add_special_tokens=False)
    assert full_ids[-1] == im_end or full_ids[-1 - len(newline) :] == [im_end, *newline]


@pytest.mark.data
def test_record_fits_max_seq_length() -> None:
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    cfg = AppConfig(run={"run_name": "t"})
    dataset, dropped = build_sft_dataset(cfg, tok)
    print(f"train records: {len(dataset)}, dropped as overlong: {dropped}")
    # 82 baseball_1 questions (max 2336 tokens) exceed 2048; the longest dev record is 1016
    assert dropped < 0.02 * (len(dataset) + dropped)
