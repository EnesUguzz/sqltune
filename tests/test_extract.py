from __future__ import annotations

from sqltune.inference.extract import extract_sql


def test_fenced_sql() -> None:
    text = "Here it is:\n```sql\nSELECT name FROM singer;\n```\nDone."
    assert extract_sql(text) == "SELECT name FROM singer"


def test_last_sql_block_wins() -> None:
    text = "```sql\nSELECT 1\n```\nfixed:\n```sql\nSELECT 2\n```"
    assert extract_sql(text) == "SELECT 2"


def test_plain_fence() -> None:
    assert extract_sql("```\nSELECT 3\n```") == "SELECT 3"


def test_plain_select() -> None:
    assert extract_sql("The answer is SELECT count(*) FROM singer") == "SELECT count(*) FROM singer"


def test_with_cte() -> None:
    assert (
        extract_sql("WITH t AS (SELECT 1) SELECT * FROM t;")
        == "WITH t AS (SELECT 1) SELECT * FROM t"
    )


def test_think_removed() -> None:
    text = "<think>maybe ```sql\nSELECT 9\n```</think>\n```sql\nSELECT 1\n```"
    assert extract_sql(text) == "SELECT 1"


def test_none_when_no_sql() -> None:
    assert extract_sql("I cannot answer that.") is None
    assert extract_sql("") is None


def test_only_first_statement() -> None:
    assert extract_sql("SELECT 1; DROP TABLE x;") == "SELECT 1"


def test_semicolon_in_string_kept() -> None:
    assert extract_sql("SELECT ';' FROM t; SELECT 2") == "SELECT ';' FROM t"
