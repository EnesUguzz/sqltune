"""Build the chat messages for one question; the single prompt used everywhere.

Purpose: Baseline, fine-tune training and every comparison model must see exactly the same
text, or a score delta mixes the model change with a prompt change.
Flow: Example + PromptConfig -> build_messages() -> runner (inference) and training/sft_data.py.
Params: PromptConfig.system_template, PromptConfig.single_template.
Try: Edit prompts/sql_single.md and diff `build_messages(ex, cfg)` before / after.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqltune.config.models import PromptConfig
from sqltune.domain import Example
from sqltune.prompting.schema import schema_text


@lru_cache(maxsize=8)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def build_messages(ex: Example, cfg: PromptConfig) -> list[dict[str, str]]:
    """[system, user] messages; the user template's {schema} and {question} are filled in."""
    user = (
        _read(cfg.single_template)
        .replace("{schema}", schema_text(ex.db_path))
        .replace("{question}", ex.question.strip())
    )
    return [
        {"role": "system", "content": _read(cfg.system_template)},
        {"role": "user", "content": user},
    ]
