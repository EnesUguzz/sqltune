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


def test_sample_rows_three(tiny_db: Path) -> None:
    text = schema_text(tiny_db, sample_rows=3)
    block = text.split("/* 3 example rows from singer:")[1].split("*/")[0]
    assert len([line for line in block.strip().splitlines() if line.strip()]) == 4  # header + 3
    assert "Ada" in block


def test_long_values_truncated(tmp_path: Path) -> None:
    import sqlite3

    db = tmp_path / "long.sqlite"
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE t (x TEXT)")
    con.execute("INSERT INTO t VALUES (?)", ("y" * 500,))
    con.commit()
    con.close()
    assert "y" * 41 not in schema_text(db, sample_rows=1)


def test_evidence_in_prompt_only_when_present(tiny_db: Path) -> None:
    ex = _example(tiny_db)
    assert "Evidence" not in build_messages(ex, PROMPT_CFG)[1]["content"]
    with_ev = ex.model_copy(update={"evidence": "France refers to country = France"})
    user = build_messages(with_ev, PROMPT_CFG)[1]["content"]
    assert "Evidence: France refers to country = France" in user


def _pool() -> list[Example]:
    return [
        Example(
            id=f"spider-train-{i:04d}",
            split="train",
            db_id=f"db{i % 5}",
            db_path=Path("unused"),
            question=f"q{i}",
            gold_sql=f"SELECT {i}",
        )
        for i in range(50)
    ]


def test_fewshot_excludes_target_db(tiny_db: Path) -> None:
    from sqltune.prompting.fewshot import pick_shots

    target = _example(tiny_db).model_copy(update={"db_id": "db2"})
    shots = pick_shots(target, _pool(), k=3, seed=13)
    assert len(shots) == 3
    assert all(s.db_id != "db2" for s in shots)


def test_fewshot_deterministic(tiny_db: Path) -> None:
    from sqltune.prompting.fewshot import pick_shots

    target = _example(tiny_db)
    assert pick_shots(target, _pool(), 3, 13) == pick_shots(target, _pool(), 3, 13)


def test_fewshot_messages_are_pairs(tiny_db: Path) -> None:
    shot = _example(tiny_db).model_copy(update={"id": "spider-train-0001", "gold_sql": "SELECT 7"})
    msgs = build_messages(_example(tiny_db), PROMPT_CFG, shots=[shot])
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert msgs[2]["content"] == "```sql\nSELECT 7\n```"


def test_retry_message_has_only_error_text(tiny_db: Path) -> None:
    from sqltune.prompting.builder import retry_messages

    first = build_messages(_example(tiny_db), PROMPT_CFG)
    msgs = retry_messages(first, "```sql\nSELECT nope\n```", "no such column: nope", PROMPT_CFG)
    assert [m["role"] for m in msgs] == ["system", "user", "assistant", "user"]
    assert "no such column: nope" in msgs[-1]["content"]
    assert "SELECT count(*)" not in msgs[-1]["content"]  # never the gold SQL
    assert "correct" not in msgs[-1]["content"].lower().replace("corrected", "")
