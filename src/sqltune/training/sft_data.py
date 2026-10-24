"""Turn Spider / BIRD train questions into prompt-completion records for supervised fine-tuning.

Purpose: The model must learn from exactly the prompt it is evaluated with; the prompt is
taken from the same build_messages() the evaluator uses, and the completion is the gold SQL
in the fenced format the system prompt asks for. TRL computes loss on the completion only.
Flow: load_spider(train) / load_bird_train() -> split_by_db (validation) -> to_sft_record()
-> build_sft_dataset() (drops overlong, counts per source and DB) -> trainer.
Params: PromptConfig (shared with eval), DataConfig.train_sources, TrainConfig.max_seq_length,
val_fraction, train_sample_rows, train_sample_rows_mix.
Try: `to_sft_record(load_spider(DataConfig(), "train")[0], PromptConfig())`.
"""

from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Protocol

from sqltune.config.models import AppConfig, PromptConfig
from sqltune.data.spider import load_spider
from sqltune.data.subsets import select_fraction, split_by_db
from sqltune.domain import Example
from sqltune.prompting.builder import build_messages

if TYPE_CHECKING:
    from datasets import Dataset


class ChatTokenizer(Protocol):
    def apply_chat_template(self, messages: list[dict[str, str]], **kwargs: Any) -> Any: ...


def to_sft_record(ex: Example, cfg: PromptConfig) -> dict[str, list[dict[str, str]]]:
    return {
        "prompt": build_messages(ex, cfg),
        "completion": [{"role": "assistant", "content": f"```sql\n{ex.gold_sql.strip()}\n```"}],
    }


def _n_tokens(tokenizer: ChatTokenizer, messages: list[dict[str, str]]) -> int:
    ids = tokenizer.apply_chat_template(messages, tokenize=True)
    if isinstance(ids, dict) or hasattr(ids, "keys"):  # transformers 5 returns a BatchEncoding
        ids = ids["input_ids"]
    return len(ids)


@dataclass
class SftData:
    train: Dataset
    val: Dataset | None
    dropped: dict[str, int]  # source -> records dropped for exceeding max_seq_length
    dropped_by_db: dict[str, int]  # db_id -> dropped records (wide schemas show up here)
    counts: dict[str, dict[str, int]]  # source -> {"train": n, "val": n} after dropping


def _load_source(cfg: AppConfig, source: str) -> list[Example]:
    if source == "spider":
        return load_spider(cfg.data, "train")
    from sqltune.data.bird_train import load_bird_train

    return load_bird_train(cfg.data)


def _rows_for(ex: Example, cfg: AppConfig) -> int:
    t = cfg.training
    if t.train_sample_rows_mix <= 0:
        return cfg.prompt.sample_rows
    draw = random.Random(f"{t.seed}-{ex.id}").random()
    return t.train_sample_rows if draw < t.train_sample_rows_mix else 0


def build_sft_dataset(cfg: AppConfig, tokenizer: ChatTokenizer) -> SftData:
    """Records per source -> nested fraction -> DB-disjoint validation split -> prompt (with
    or without example rows) -> drop records longer than max_seq_length."""
    from datasets import Dataset

    t = cfg.training
    out: dict[str, list[dict[str, Any]]] = {"train": [], "val": []}
    dropped: dict[str, int] = {}
    dropped_by_db: Counter[str] = Counter()
    counts: dict[str, dict[str, int]] = {}
    for source in cfg.data.train_sources:
        examples = select_fraction(_load_source(cfg, source), t.train_fraction, t.seed)
        train, val = split_by_db(examples, t.val_fraction, t.seed)
        dropped[source] = 0
        counts[source] = {}
        for part, items in (("train", train), ("val", val)):
            kept = 0
            for ex in items:
                prompt_cfg = cfg.prompt.model_copy(update={"sample_rows": _rows_for(ex, cfg)})
                record = to_sft_record(ex, prompt_cfg)
                if _n_tokens(tokenizer, record["prompt"] + record["completion"]) > t.max_seq_length:
                    dropped[source] += 1
                    dropped_by_db[ex.db_id] += 1
                    continue
                out[part].append(record)
                kept += 1
            counts[source][part] = kept
    if len(cfg.data.train_sources) > 1:
        random.Random(t.seed).shuffle(out["train"])
    return SftData(
        train=Dataset.from_list(out["train"]),
        val=Dataset.from_list(out["val"]) if out["val"] else None,
        dropped=dropped,
        dropped_by_db=dict(dropped_by_db),
        counts=counts,
    )
