from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel, ValidationError

from sqltune.config.loader import load_config
from sqltune.config.models import AppConfig


def test_defaults() -> None:
    cfg = AppConfig(run={"run_name": "x"})
    assert cfg.training.r == 16
    assert cfg.inference.model == "Qwen/Qwen2.5-Coder-3B-Instruct"
    assert cfg.data.subset == "full"


def test_extends_merges(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "run: {run_name: base}\ntraining: {r: 32, lora_alpha: 32}\n", encoding="utf-8"
    )
    (tmp_path / "child.yaml").write_text(
        "extends: base.yaml\nrun: {run_name: child}\ntraining: {r: 8}\n", encoding="utf-8"
    )
    cfg = load_config(tmp_path / "child.yaml")
    assert cfg.run.run_name == "child"
    assert cfg.training.r == 8
    assert cfg.training.lora_alpha == 32


def test_unknown_key_rejected(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "run: {run_name: x}\ntraining: {rank: 8}\n", encoding="utf-8"
    )
    with pytest.raises(ValidationError):
        load_config(tmp_path / "bad.yaml")


def test_circular_extends_rejected(tmp_path: Path) -> None:
    (tmp_path / "a.yaml").write_text("extends: b.yaml\n", encoding="utf-8")
    (tmp_path / "b.yaml").write_text("extends: a.yaml\n", encoding="utf-8")
    with pytest.raises(ValueError, match="circular"):
        load_config(tmp_path / "a.yaml")


def _all_fields(model: type[BaseModel]) -> list[tuple[str, str | None]]:
    out: list[tuple[str, str | None]] = []
    for name, field in model.model_fields.items():
        ann = field.annotation
        if isinstance(ann, type) and issubclass(ann, BaseModel):
            out.extend(_all_fields(ann))
        else:
            out.append((f"{model.__name__}.{name}", field.description))
    return out


def test_every_field_documented() -> None:
    undocumented = [n for n, d in _all_fields(AppConfig) if not (d or "").startswith("What:")]
    assert undocumented == []


def test_base_yaml_loads() -> None:
    cfg = load_config(Path(__file__).parents[1] / "experiments/configs/base.yaml")
    assert cfg.run.experiment_name == "m0"
