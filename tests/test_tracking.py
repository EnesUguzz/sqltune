from __future__ import annotations

import json
from pathlib import Path

from sqltune.config.models import AppConfig
from sqltune.evaluation.tracking import run_tags


def _cfg(tmp_path: Path, model: str, run_name: str) -> AppConfig:
    return AppConfig(
        run={"run_name": run_name, "results_dir": tmp_path}, inference={"model": model}
    )


def test_tags_link_adapter_eval_to_training_run(tmp_path: Path) -> None:
    (tmp_path / "lc_f1").mkdir()
    (tmp_path / "lc_f1" / "train_summary.json").write_text(
        json.dumps({"mlflow_run_id": "abc"}), encoding="utf-8"
    )
    tags = run_tags(_cfg(tmp_path, "lc_f1", "m1_lc_f1__zero__birddiag"))
    assert tags == {
        "train_run": "lc_f1",
        "train_mlflow_run_id": "abc",
        "prompt_variant": "zero",
        "eval_set": "birddiag",
    }


def test_tags_for_base_model(tmp_path: Path) -> None:
    assert run_tags(_cfg(tmp_path, "base", "m0_baseline")) == {}
