from __future__ import annotations

import json
from pathlib import Path

import pytest

from sqltune.config.models import AppConfig, DataConfig, PromptConfig
from sqltune.evaluation.metrics import compute_metrics, mcnemar_exact_p
from sqltune.inference.openai_compat import Completion
from sqltune.runner import run_eval

REPO = Path(__file__).parents[1]


class FakeBackend:
    """Answers question i with a fixed SQL from `answers` (cycled)."""

    def __init__(self, answers: list[str]) -> None:
        self.answers = answers
        self.calls = 0

    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        out = []
        for _ in requests:
            text = self.answers[self.calls % len(self.answers)]
            out.append(Completion(text=text, latency_ms=10.0, prompt_tokens=5, completion_tokens=2))
            self.calls += 1
        return out


def _cfg(data: DataConfig, tmp_path: Path, run_name: str, ref: str | None = None) -> AppConfig:
    return AppConfig(
        data=data,
        prompt=PromptConfig(
            system_template=REPO / "prompts/sql_system.md",
            single_template=REPO / "prompts/sql_single.md",
        ),
        evaluation={"mlflow_tracking_uri": f"sqlite:///{tmp_path}/mlflow.db", "exec_workers": 2},
        run={"run_name": run_name, "results_dir": tmp_path / "results", "reference_run": ref},
    )


GOOD = "```sql\nSELECT count(*) FROM singer\n```"
BAD = "```sql\nSELECT count(*) FROM concert WHERE year = 2015\n```"


def test_run_writes_artifacts(tiny_spider: DataConfig, tmp_path: Path) -> None:
    out = run_eval(_cfg(tiny_spider, tmp_path, "r1"), backend=FakeBackend([GOOD, BAD, "no idea"]))
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(rows) == 6
    assert {r["outcome"] for r in rows} == {"correct", "wrong_result", "no_sql"}
    assert "col_tolerant_correct" in rows[0]
    metrics = json.loads((out / "metrics.json").read_text())
    assert metrics["n"] == 6
    assert metrics["ex"] == pytest.approx(2 / 6)
    assert (out / "report.md").exists()
    assert (out / "config.yaml").exists()


def _row(id_: str, outcome: str) -> dict[str, object]:
    return {
        "id": id_,
        "outcome": outcome,
        "col_tolerant_correct": outcome == "correct",
        "latency_ms": 1.0,
        "prompt_tokens": 1,
        "completion_tokens": 1,
    }


def test_ex_excludes_gold_errors() -> None:
    m = compute_metrics([_row("a", "correct"), _row("b", "gold_error"), _row("c", "exec_error")])
    assert m["n"] == 3
    assert m["n_gold_error"] == 1
    assert m["ex"] == pytest.approx(0.5)
    assert m["rate_exec_error"] == pytest.approx(0.5)


def test_transport_errors_counted() -> None:
    """A request that failed after retries is a server problem, not the model's no_sql."""
    failed = {**_row("b", "no_sql"), "error": "APIConnectionError()"}
    m = compute_metrics([_row("a", "correct"), failed])
    assert m["n_transport_error"] == 1


def test_mcnemar_exact_p() -> None:
    assert mcnemar_exact_p(0, 0) == 1.0
    assert mcnemar_exact_p(10, 0) == pytest.approx(0.001953125)
    assert mcnemar_exact_p(5, 5) == 1.0


def test_report_shows_delta_vs_reference(tiny_spider: DataConfig, tmp_path: Path) -> None:
    run_eval(_cfg(tiny_spider, tmp_path, "ref"), backend=FakeBackend([BAD]))
    out = run_eval(_cfg(tiny_spider, tmp_path, "new", ref="ref"), backend=FakeBackend([GOOD]))
    report = (out / "report.md").read_text()
    assert "vs ref" in report
    assert "+100.0" in report  # EX 0% -> 100%
    assert "fixed: 6" in report
    assert "broken: 0" in report
    assert "McNemar" in report
