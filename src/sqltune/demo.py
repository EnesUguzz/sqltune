"""Gradio demo: ask a question about a SQLite database, get the model's SQL and its result.

Purpose: Lets anyone try the fine-tuned model on their own questions. The model never needs
a hand-written schema: the app reads it from the .sqlite file with the same schema_text()
used in training and evaluation, builds the same prompt (build_messages), and runs the
answer read-only with the evaluator's timeout and row cap.
Flow: pick a bundled database (Spider dev / test, BIRD Mini-Dev) or upload a .sqlite ->
schema shown -> question (+ optional BIRD-style evidence hint) -> served model (vLLM, any
name it serves, e.g. base or m1b_mix) -> extract_sql -> execute() -> SQL + result table;
on an execution error the model gets one retry with the error text (as in rows3 + retry).
Params: --config (inference / prompt / evaluation settings, typical
experiments/configs/base.yaml; data_dir comes from it), --port, --share. Model choices come
from the server's /v1/models.
Try: `make serve-multi ADAPTERS="m1b_mix"` then `make demo` and open http://localhost:7860.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

from sqltune.config.loader import load_config
from sqltune.config.models import AppConfig
from sqltune.domain import Example
from sqltune.evaluation.executor import execute
from sqltune.inference.extract import extract_sql
from sqltune.inference.openai_compat import OpenAICompatBackend
from sqltune.prompting.builder import build_messages, retry_messages
from sqltune.prompting.schema import schema_text

MAX_SHOWN_ROWS = 50


def list_databases(data_dir: Path) -> dict[str, Path]:
    """'<source>/<db_id>' -> .sqlite path: Spider (train + dev), Spider test, BIRD Mini-Dev."""
    roots = {
        "spider": data_dir / "spider" / "database",
        "spider_test": data_dir / "spider" / "test_database",
        "bird": data_dir / "bird" / "dev_databases",
    }
    found: dict[str, Path] = {}
    for source, root in roots.items():
        for path in sorted(root.glob("*/*.sqlite")) if root.exists() else []:
            found.setdefault(f"{source}/{path.parent.name}", path)
    return found


def answer(
    question: str,
    db_path: Path,
    model: str,
    cfg: AppConfig,
    evidence: str = "",
    backend: Any = None,
) -> dict[str, Any]:
    """SQL, column names, rows (first MAX_SHOWN_ROWS), total row count and error, if any."""
    ex = Example(
        id="demo",
        split="demo",
        db_id=db_path.stem,
        db_path=db_path,
        question=question,
        gold_sql="",
        evidence=evidence.strip() or None,
    )
    inference = cfg.inference.model_copy(update={"model": model})
    backend = backend or OpenAICompatBackend(inference)
    messages = build_messages(ex, cfg.prompt)
    [first] = backend.complete_many([messages])
    if first.error:
        return {
            "sql": "",
            "columns": [],
            "rows": [],
            "n_rows": 0,
            "error": first.error,
            "retried": False,
        }
    sql = extract_sql(first.text)
    result = execute(db_path, sql, cfg.evaluation) if sql else None
    retried = False
    if result is not None and not result.ok and result.error_type == "exec_error":
        [second] = backend.complete_many(
            [retry_messages(messages, first.text, result.error or "", cfg.prompt)]
        )
        sql = extract_sql(second.text) or sql
        assert sql is not None  # a result exists, so the first answer had SQL
        result = execute(db_path, sql, cfg.evaluation)
        retried = True
    if result is None:
        return {
            "sql": first.text,
            "columns": [],
            "rows": [],
            "n_rows": 0,
            "error": "no SQL in the answer",
            "retried": retried,
        }
    rows = result.rows or []
    return {
        "sql": sql,
        "columns": result.columns or [],
        "rows": [list(r) for r in rows[:MAX_SHOWN_ROWS]],
        "n_rows": len(rows),
        "error": None if result.ok else result.error,
        "retried": retried,
    }


def served_models(cfg: AppConfig) -> list[str]:
    import httpx

    try:
        resp = httpx.get(f"{cfg.inference.base_url}/models", timeout=5)
        return [m["id"] for m in resp.json()["data"]]
    except (httpx.HTTPError, KeyError, ValueError):
        return [cfg.inference.model]


def build_app(cfg: AppConfig) -> Any:
    import gradio as gr

    dbs = list_databases(cfg.data.data_dir)
    models = served_models(cfg)
    default_model = "m1b_mix" if "m1b_mix" in models else models[0]

    def pick_path(choice: str | None, upload: str | None) -> Path | None:
        if upload:
            return Path(upload)
        return dbs.get(choice or "")

    def show_schema(choice: str | None, upload: str | None) -> str:
        path = pick_path(choice, upload)
        return schema_text(path) if path else ""

    def run(
        question: str, evidence: str, choice: str | None, upload: str | None, model: str
    ) -> tuple[str, Any, str]:
        path = pick_path(choice, upload)
        if path is None or not question.strip():
            return "", None, "Pick a database and type a question."
        out = answer(question, path, model, cfg, evidence)
        note = f"{out['n_rows']} rows" + (
            f" (first {MAX_SHOWN_ROWS} shown)" if out["n_rows"] > MAX_SHOWN_ROWS else ""
        )
        if out["retried"]:
            note += " · retried once after an execution error"
        if out["error"]:
            note = f"Error: {out['error']}"
        table = {"headers": out["columns"], "data": out["rows"]} if out["columns"] else None
        return f"```sql\n{out['sql']}\n```", table, note

    with gr.Blocks(title="sqltune: text-to-SQL demo") as app:
        gr.Markdown(
            "## sqltune: ask a SQLite database in plain English\n"
            "Qwen2.5-Coder-3B fine-tuned with QLoRA on Spider + BIRD train. The schema is read "
            "from the database file; the query runs read-only."
        )
        with gr.Row():
            with gr.Column(scale=1):
                db = gr.Dropdown(
                    sorted(dbs),
                    label="Database",
                    value="spider/concert_singer" if "spider/concert_singer" in dbs else None,
                )
                upload = gr.File(
                    label="...or upload a .sqlite file",
                    file_types=[".sqlite", ".db"],
                    type="filepath",
                )
                model = gr.Dropdown(models, value=default_model, label="Model")
                schema = gr.Code(label="Schema", language="sql", lines=14)
            with gr.Column(scale=1):
                question = gr.Textbox(
                    label="Question", placeholder="How many singers are older than 30?"
                )
                evidence = gr.Textbox(
                    label="Hint (optional, BIRD-style evidence)",
                    placeholder="older refers to age > 30",
                )
                go = gr.Button("Ask", variant="primary")
                sql = gr.Markdown()
                note = gr.Markdown()
                table = gr.Dataframe(label="Result", wrap=True)
        for trigger in (db.change, upload.change, app.load):
            trigger(show_schema, [db, upload], schema)
        go.click(run, [question, evidence, db, upload, model], [sql, table, note])
        question.submit(run, [question, evidence, db, upload, model], [sql, table, note])
    return app


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", default="experiments/configs/base.yaml")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    build_app(load_config(args.config)).launch(server_port=args.port, share=args.share)


if __name__ == "__main__":
    main()
