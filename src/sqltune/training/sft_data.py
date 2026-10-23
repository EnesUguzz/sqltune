"""Turn Spider train questions into prompt-completion records for supervised fine-tuning.

Purpose: The model must learn from exactly the prompt it is evaluated with; the prompt is
taken from the same build_messages() the evaluator uses, and the completion is the gold SQL
in the fenced format the system prompt asks for. TRL computes loss on the completion only.
Flow: load_spider(train) -> to_sft_record() -> build_sft_dataset() (drops overlong) -> trainer.
Params: PromptConfig (shared with eval), TrainConfig.max_seq_length.
Try: `to_sft_record(load_spider(DataConfig(), "train")[0], PromptConfig())`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Protocol

from sqltune.config.models import AppConfig, PromptConfig
from sqltune.data.spider import load_spider
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


def build_sft_dataset(cfg: AppConfig, tokenizer: ChatTokenizer) -> tuple[Dataset, int]:
    """(train dataset, number of records dropped for exceeding max_seq_length)."""
    from datasets import Dataset

    records = []
    dropped = 0
    for ex in load_spider(cfg.data, "train"):
        record = to_sft_record(ex, cfg.prompt)
        if (
            _n_tokens(tokenizer, record["prompt"] + record["completion"])
            > cfg.training.max_seq_length
        ):
            dropped += 1
            continue
        records.append(record)
    return Dataset.from_list(records), dropped
