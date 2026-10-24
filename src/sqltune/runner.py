"""Evaluate one model configuration end to end and write the run folder.

Purpose: The single path every comparison row goes through (same questions, same prompt,
same scorer), so differences between runs come only from what the config changes.
Flow: load_examples (Spider or BIRD) -> build_messages (+ few-shot) -> backend.complete_many
-> extract_sql -> score_example (process pool) -> [retry failed queries once] ->
results/<run_name>/{predictions.jsonl, metrics.json, config.yaml, report.md} -> MLflow.
Params: the whole AppConfig; evaluation.exec_workers processes score SQL in parallel.
Try: `make serve && make eval CONFIG=experiments/configs/m0/baseline.yaml`.
"""

from __future__ import annotations

import json
import multiprocessing
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Protocol

import structlog
import yaml

from sqltune.config.models import AppConfig, EvalConfig
from sqltune.data.bird import load_bird
from sqltune.data.spider import load_spider
from sqltune.data.subsets import select_subset
from sqltune.domain import Example
from sqltune.evaluation.executor import score_example
from sqltune.evaluation.metrics import compute_metrics
from sqltune.evaluation.report import load_rows, render_report
from sqltune.evaluation.tracking import log_to_mlflow
from sqltune.inference.extract import extract_sql
from sqltune.inference.openai_compat import Completion, OpenAICompatBackend
from sqltune.prompting.builder import build_messages, retry_messages
from sqltune.prompting.fewshot import pick_shots

log = structlog.get_logger()


class Backend(Protocol):
    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]: ...


def _score(args: tuple[Example, str | None, EvalConfig]) -> tuple[str, bool, str | None]:
    ex, pred_sql, cfg = args
    outcome, tolerant, result = score_example(ex, pred_sql, cfg)
    return outcome, tolerant, result.error if result is not None else None


def _score_all(
    examples: list[Example], preds: list[str | None], cfg: EvalConfig
) -> list[tuple[str, bool, str | None]]:
    jobs = [(ex, sql, cfg) for ex, sql in zip(examples, preds, strict=True)]
    if not jobs:
        return []
    # spawn: the parent holds asyncio/HTTP threads, which fork would copy in a broken state
    ctx = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=min(cfg.exec_workers, len(jobs)), mp_context=ctx) as pool:
        return list(pool.map(_score, jobs, chunksize=16))


class TestSetGuardError(RuntimeError):
    """A test set (Spider test, bird_holdout) was requested by a config without run.test_once."""

    __test__ = False  # not a pytest test class despite the name


def _is_test_set(cfg: AppConfig) -> bool:
    if cfg.data.dataset == "bird":
        return cfg.data.bird_split == "holdout"
    return cfg.data.split == "test"


def load_examples(cfg: AppConfig) -> list[Example]:
    if _is_test_set(cfg) and not cfg.run.test_once:
        raise TestSetGuardError(
            f"{cfg.run.run_name}: test sets are run once per final model; "
            "set run.test_once: true in a final config"
        )
    if cfg.data.dataset == "bird":
        return load_bird(cfg.data)
    return select_subset(load_spider(cfg.data, cfg.data.split), cfg.data)


def build_requests(examples: list[Example], cfg: AppConfig) -> list[list[dict[str, str]]]:
    if cfg.prompt.few_shot_k == 0:
        return [build_messages(ex, cfg.prompt) for ex in examples]
    pool = load_spider(cfg.data.model_copy(update={"include_train_others": False}), "train")
    return [
        build_messages(
            ex, cfg.prompt, pick_shots(ex, pool, cfg.prompt.few_shot_k, cfg.prompt.few_shot_seed)
        )
        for ex in examples
    ]


def run_eval(cfg: AppConfig, backend: Backend | None = None) -> Path:
    examples = load_examples(cfg)
    backend = backend or OpenAICompatBackend(cfg.inference)
    out = cfg.run.results_dir / cfg.run.run_name
    out.mkdir(parents=True, exist_ok=True)

    requests = build_requests(examples, cfg)
    start = time.monotonic()
    completions = backend.complete_many(requests)
    preds = [extract_sql(c.text) for c in completions]
    scores = _score_all(examples, preds, cfg.evaluation)
    retried = [False] * len(examples)
    first_sql: list[str | None] = [None] * len(examples)

    failed = [i for i, (outcome, _, _) in enumerate(scores) if outcome == "exec_error"]
    if cfg.inference.retry_on_error and failed:
        second = backend.complete_many(
            [
                retry_messages(requests[i], completions[i].text, scores[i][2] or "", cfg.prompt)
                for i in failed
            ]
        )
        second_preds = [extract_sql(c.text) for c in second]
        second_scores = _score_all([examples[i] for i in failed], second_preds, cfg.evaluation)
        for j, i in enumerate(failed):
            first, again = completions[i], second[j]
            completions[i] = Completion(
                text=again.text,
                latency_ms=first.latency_ms + again.latency_ms,
                prompt_tokens=(first.prompt_tokens or 0) + (again.prompt_tokens or 0),
                completion_tokens=(first.completion_tokens or 0) + (again.completion_tokens or 0),
                error=again.error,
            )
            first_sql[i], preds[i], scores[i] = preds[i], second_preds[j], second_scores[j]
            retried[i] = True
    gen_s = time.monotonic() - start
    log.info("scored", n=len(examples), retried=sum(retried), generation_s=round(gen_s, 1))

    rows: list[dict[str, Any]] = []
    for ex, comp, pred, (outcome, tolerant, pred_error), was_retried, first in zip(
        examples, completions, preds, scores, retried, first_sql, strict=True
    ):
        rows.append(
            {
                "id": ex.id,
                "db_id": ex.db_id,
                "question": ex.question,
                "gold_sql": ex.gold_sql,
                "raw_text": comp.text,
                "pred_sql": pred,
                "outcome": outcome,
                "col_tolerant_correct": tolerant,
                "pred_error": pred_error,
                "retried": was_retried,
                "first_pred_sql": first,
                "difficulty": ex.difficulty,
                "latency_ms": comp.latency_ms,
                "prompt_tokens": comp.prompt_tokens,
                "completion_tokens": comp.completion_tokens,
                "error": comp.error,
            }
        )
    with (out / "predictions.jsonl").open("w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")

    metrics = compute_metrics(rows)
    metrics["generation_s"] = gen_s
    (out / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (out / "config.yaml").write_text(
        yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False), encoding="utf-8"
    )

    reference = None
    if cfg.run.reference_run:
        ref_dir = cfg.run.results_dir / cfg.run.reference_run
        ref_metrics = json.loads((ref_dir / "metrics.json").read_text(encoding="utf-8"))
        reference = (cfg.run.reference_run, load_rows(ref_dir), ref_metrics)
    (out / "report.md").write_text(
        render_report(cfg.run.run_name, rows, metrics, reference), encoding="utf-8"
    )
    log_to_mlflow(cfg, metrics, out)
    return out
