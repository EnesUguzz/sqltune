"""Build the chat messages for one question; the single prompt used everywhere.

Purpose: Baseline, fine-tune training and every comparison model must see exactly the same
text, or a score delta mixes the model change with a prompt change. Prompt variants
(example rows, few-shot, retry) are explicit config switches, reported as their own rows.
Flow: Example + PromptConfig (+ few-shot examples) -> build_messages() -> runner / SFT data;
after an execution error -> retry_messages() -> one more model turn.
Params: PromptConfig templates, sample_rows; evidence is added when the example has one (BIRD).
Try: Edit prompts/sql_single.md and diff `build_messages(ex, cfg)` before / after.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from sqltune.config.models import PromptConfig
from sqltune.domain import Example
from sqltune.prompting.schema import schema_text


@lru_cache(maxsize=16)
def _read(path: Path) -> str:
    return path.read_text(encoding="utf-8").strip()


def _fenced(sql: str) -> str:
    return f"```sql\n{sql.strip()}\n```"


def user_message(ex: Example, cfg: PromptConfig) -> str:
    template = cfg.evidence_template if ex.evidence else cfg.single_template
    return (
        _read(template)
        .replace("{schema}", schema_text(ex.db_path, cfg.sample_rows))
        .replace("{evidence}", (ex.evidence or "").strip())
        .replace("{question}", ex.question.strip())
    )


def build_messages(
    ex: Example, cfg: PromptConfig, shots: list[Example] | None = None
) -> list[dict[str, str]]:
    """[system, (user, assistant) per shot, user]; shots are solved examples from other DBs."""
    messages = [{"role": "system", "content": _read(cfg.system_template)}]
    for shot in shots or []:
        messages += [
            {"role": "user", "content": user_message(shot, cfg)},
            {"role": "assistant", "content": _fenced(shot.gold_sql)},
        ]
    messages.append({"role": "user", "content": user_message(ex, cfg)})
    return messages


def retry_messages(
    messages: list[dict[str, str]], first_answer: str, error: str, cfg: PromptConfig
) -> list[dict[str, str]]:
    """The first conversation plus the model's answer and the SQLite error text only; the
    model never learns whether a query that ran was right (no gold leakage)."""
    feedback = _read(cfg.retry_template).replace("{error}", error.strip())
    return [
        *messages,
        {"role": "assistant", "content": first_answer},
        {"role": "user", "content": feedback},
    ]
