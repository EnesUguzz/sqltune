from __future__ import annotations

import json
from pathlib import Path

import yaml

from sqltune.evaluation.summary import build_summary


def _run(
    root: Path, name: str, model: str, dataset: str, hits: list[bool], **extra: object
) -> None:
    out = root / name
    out.mkdir(parents=True)
    rows = [
        {
            "id": f"q{i}",
            "db_id": f"db{i % 2}",
            "outcome": "correct" if h else "exec_error",
            "col_tolerant_correct": h,
            "pred_error": None if h else "no such column: x",
            "retried": False,
        }
        for i, h in enumerate(hits)
    ]
    (out / "predictions.jsonl").write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    ex = sum(hits) / len(hits)
    metrics = {
        "n": len(hits),
        "ex": ex,
        "ex_col_tolerant": ex,
        "rate_exec_error": 1 - ex,
        "latency_p50_ms": 100.0,
        "n_transport_error": 0,
    }
    (out / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    cfg = {
        "inference": {"model": model, "retry_on_error": False},
        "prompt": {"sample_rows": 0, "few_shot_k": 0},
        "data": {"dataset": dataset},
        **extra,
    }
    (out / "config.yaml").write_text(yaml.safe_dump(cfg), encoding="utf-8")


def test_summary_rows_and_delta(tmp_path: Path) -> None:
    _run(tmp_path, "m1_base__zero__spider", "base", "spider", [True, False, False, False])
    _run(tmp_path, "m1_m0_qlora__zero__spider", "m0_qlora", "spider", [True, True, True, False])
    _run(tmp_path, "m1_base__zero__birddiag", "base", "bird", [True, True])
    text = build_summary(tmp_path)
    assert "| spider |" in text and "| birddiag |" in text
    line = next(x for x in text.splitlines() if x.startswith("| m0_qlora | zero | spider"))
    assert "75.0" in line  # EX
    assert "+50.0" in line  # vs base zero-shot on the same set
    assert "no such column" in text or "nsc" in text


def test_summary_ignores_non_m1_runs(tmp_path: Path) -> None:
    _run(tmp_path, "m0_baseline", "base", "spider", [True])
    _run(tmp_path, "m1_base__zero__spider", "base", "spider", [True, False])
    assert "m0_baseline" not in build_summary(tmp_path)
