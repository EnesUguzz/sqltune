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
    assert build_sft_dataset(cfg, CharTokenizer()).dropped == {"spider": 0}
    short = cfg.model_copy(
        update={"training": cfg.training.model_copy(update={"max_seq_length": 256})}
    )
    data = build_sft_dataset(short, CharTokenizer())
    assert data.dropped == {"spider": 4}  # every tiny record is > 256 chars with the schema
    assert len(data.train) == 0


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
    data = build_sft_dataset(cfg, tok)
    dataset, dropped = data.train, data.dropped["spider"]
    print(f"train records: {len(dataset)}, dropped as overlong: {dropped}")
    # 82 baseball_1 questions (max 2336 tokens) exceed 2048; the longest dev record is 1016
    assert dropped < 0.02 * (len(dataset) + dropped)


def _mixed_cfg(tmp_path: Path, **training: Any) -> AppConfig:
    """Spider with 4 DBs x 3 questions plus BIRD train with 4 DBs x 3 questions."""
    import json

    from conftest import make_tiny_db
    from test_bird_train import make_bird_train

    data = tmp_path / "data"
    spider = data / "spider"
    records = []
    for d in range(4):
        make_tiny_db(spider / "database" / f"sp{d}" / f"sp{d}.sqlite")
        records += [
            {"db_id": f"sp{d}", "question": f"Q{q}?", "query": "SELECT count(*) FROM singer"}
            for q in range(3)
        ]
    (spider / "train_spider.json").write_text(json.dumps(records), encoding="utf-8")
    make_bird_train(data)
    return AppConfig(
        data=DataConfig(data_dir=data, train_sources=["spider", "bird"]),
        prompt=PROMPT_CFG.model_copy(
            update={"evidence_template": REPO / "prompts/sql_single_evidence.md"}
        ),
        run={"run_name": "t"},
        training=training,
    )


def _has_rows(record: dict[str, Any]) -> bool:
    return "example rows from" in record["prompt"][-1]["content"]


def test_sources_mixed_with_db_disjoint_val(tmp_path: Path) -> None:
    pytest.importorskip("datasets", reason="needs the train extra: uv sync --extra train")
    data = build_sft_dataset(_mixed_cfg(tmp_path, val_fraction=0.2), CharTokenizer())
    assert data.counts["spider"]["train"] + data.counts["spider"]["val"] == 12
    assert data.counts["bird"]["val"] > 0 and data.counts["spider"]["val"] > 0
    assert data.val is not None and len(data.train) + len(data.val) == 24


def test_rows_mix_assigns_both_prompts(tmp_path: Path) -> None:
    pytest.importorskip("datasets", reason="needs the train extra: uv sync --extra train")
    none = build_sft_dataset(_mixed_cfg(tmp_path / "a"), CharTokenizer())
    assert not any(_has_rows(r) for r in none.train)
    mixed = build_sft_dataset(
        _mixed_cfg(tmp_path / "b", train_sample_rows_mix=0.5), CharTokenizer()
    )
    flags = [_has_rows(r) for r in mixed.train]
    assert any(flags) and not all(flags)
    again = build_sft_dataset(
        _mixed_cfg(tmp_path / "c", train_sample_rows_mix=0.5), CharTokenizer()
    )
    assert [_has_rows(r) for r in again.train] == flags


def test_dropped_counted_per_source_and_db(tmp_path: Path) -> None:
    pytest.importorskip("datasets", reason="needs the train extra: uv sync --extra train")
    data = build_sft_dataset(_mixed_cfg(tmp_path, max_seq_length=256), CharTokenizer())
    assert data.dropped == {"spider": 12, "bird": 12}
    assert data.dropped_by_db["bt0"] == 3


def test_spider_only_default_unchanged(tiny_spider: DataConfig) -> None:
    pytest.importorskip("datasets", reason="needs the train extra: uv sync --extra train")
    from sqltune.data.spider import load_spider

    cfg = AppConfig(data=tiny_spider, prompt=PROMPT_CFG, run={"run_name": "t"})
    data = build_sft_dataset(cfg, CharTokenizer())
    expected = [to_sft_record(ex, PROMPT_CFG) for ex in load_spider(tiny_spider, "train")]
    assert sorted(map(str, data.train)) == sorted(map(str, expected))
    assert data.val is None
