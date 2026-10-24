from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import make_tiny_db

from sqltune.config.models import AppConfig, DataConfig, PromptConfig
from sqltune.evaluation.metrics import compute_metrics, mcnemar_exact_p
from sqltune.inference.openai_compat import Completion
from sqltune.runner import TestSetGuardError, load_examples, run_eval

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


def test_pred_error_written(tiny_spider: DataConfig, tmp_path: Path) -> None:
    bad = "```sql\nSELECT nope FROM singer\n```"
    out = run_eval(_cfg(tiny_spider, tmp_path, "err"), backend=FakeBackend([bad]))
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert all("no such column" in (r["pred_error"] or "") for r in rows)
    report = (out / "report.md").read_text()
    assert "no such column" in report
    assert "## Per database" in report


def test_cluster_ci_contains_point_estimate() -> None:
    from sqltune.evaluation.metrics import db_cluster_bootstrap_ci

    ref = [
        {"id": f"q{i}", "db_id": f"db{i % 4}", "col_tolerant_correct": i % 3 == 0}
        for i in range(40)
    ]
    new = [
        {**r, "col_tolerant_correct": r["col_tolerant_correct"] or i % 2 == 0}
        for i, r in enumerate(ref)
    ]
    delta = (
        sum(r["col_tolerant_correct"] for r in new) - sum(r["col_tolerant_correct"] for r in ref)
    ) / 40
    lo, hi = db_cluster_bootstrap_ci(ref, new, n=500, seed=0)
    assert lo <= delta <= hi
    assert db_cluster_bootstrap_ci(ref, new, n=500, seed=0) == (lo, hi)


class ScriptedBackend:
    """Returns replies[i] for the i-th call of complete_many (one list per call)."""

    def __init__(self, replies: list[list[str]]) -> None:
        self.replies = replies
        self.requests: list[list[list[dict[str, str]]]] = []

    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        texts = self.replies[len(self.requests)]
        self.requests.append(requests)
        return [
            Completion(text=texts[i % len(texts)], latency_ms=1.0) for i in range(len(requests))
        ]


def test_retry_only_after_exec_error(tiny_spider: DataConfig, tmp_path: Path) -> None:
    cfg = _cfg(tiny_spider, tmp_path, "retry")
    cfg = cfg.model_copy(
        update={"inference": cfg.inference.model_copy(update={"retry_on_error": True})}
    )
    broken = "```sql\nSELECT count(*) FROM singr\n```"
    backend = ScriptedBackend([[GOOD, broken], [GOOD]])
    out = run_eval(cfg, backend=backend)
    rows = [json.loads(line) for line in (out / "predictions.jsonl").read_text().splitlines()]
    assert len(backend.requests) == 2
    assert len(backend.requests[1]) == 3  # only the 3 exec_error questions are retried
    assert "no such table: singr" in backend.requests[1][0][-1]["content"]
    assert [r["retried"] for r in rows] == [False, True] * 3
    assert all(r["outcome"] == "correct" for r in rows)


def test_test_sets_need_test_once(tiny_spider: DataConfig, tmp_path: Path) -> None:
    root = tiny_spider.data_dir / "spider"
    make_tiny_db(root / "test_database" / "tiny" / "tiny.sqlite")
    (root / "test.json").write_text((root / "dev.json").read_text())
    cfg = _cfg(tiny_spider.model_copy(update={"split": "test"}), tmp_path, "t1")
    with pytest.raises(TestSetGuardError):
        load_examples(cfg)
    holdout = _cfg(
        tiny_spider.model_copy(update={"dataset": "bird", "bird_split": "holdout"}), tmp_path, "t2"
    )
    with pytest.raises(TestSetGuardError):
        load_examples(holdout)
    every_bird = _cfg(
        tiny_spider.model_copy(update={"dataset": "bird", "bird_split": "all"}), tmp_path, "t3"
    )
    with pytest.raises(TestSetGuardError):
        load_examples(every_bird)
    cfg.run.test_once = True
    assert len(load_examples(cfg)) == 6
