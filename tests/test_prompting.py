from __future__ import annotations

from pathlib import Path

from sqltune.config.models import PromptConfig
from sqltune.domain import Example
from sqltune.prompting.builder import build_messages
from sqltune.prompting.schema import schema_text

REPO = Path(__file__).parents[1]
PROMPT_CFG = PromptConfig(
    system_template=REPO / "prompts/sql_system.md",
    single_template=REPO / "prompts/sql_single.md",
)


def test_schema_has_create_statements(tiny_db: Path) -> None:
    text = schema_text(tiny_db)
    assert "CREATE TABLE singer" in text
    assert "CREATE TABLE concert" in text


def test_schema_sorted_and_excludes_sqlite_tables(tiny_db: Path) -> None:
    text = schema_text(tiny_db)
    assert text.index("CREATE TABLE concert") < text.index("CREATE TABLE singer")
    assert "sqlite_" not in text


def _example(db: Path) -> Example:
    return Example(
        id="spider-dev-0000",
        split="dev",
        db_id="tiny",
        db_path=db,
        question="How many singers are from France?",
        gold_sql="SELECT count(*) FROM singer WHERE country = 'France'",
    )


def test_messages_roles(tiny_db: Path) -> None:
    msgs = build_messages(_example(tiny_db), PROMPT_CFG)
    assert [m["role"] for m in msgs] == ["system", "user"]


def test_messages_contain_schema_and_question(tiny_db: Path) -> None:
    msgs = build_messages(_example(tiny_db), PROMPT_CFG)
    assert "expert SQLite developer" in msgs[0]["content"]
    user = msgs[1]["content"]
    assert user.startswith("Database schema:\nCREATE TABLE concert")
    assert user.endswith("\n\nQuestion: How many singers are from France?")
    assert "{schema}" not in user
