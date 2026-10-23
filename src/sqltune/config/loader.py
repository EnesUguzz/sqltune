"""Load experiment YAML files into AppConfig, with `extends:` inheritance.

Purpose: Lets each experiment file state only what differs from its parent (usually
base.yaml), so the diff of two YAML files is exactly the experiment's change.
Flow: experiments/configs/*.yaml -> load_config() -> AppConfig -> CLI / runner / trainer.
Params: none of its own; the top-level `extends` key names the parent file (relative path).
Try: Chain extends (base -> m0/qlora -> m0/smoke) and print `load_config(...)`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

from sqltune.config.models import AppConfig

EXTENDS_KEY = "extends"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """Recursively merge override into base. An empty mapping (`{}`) in the override
    replaces the inherited one, so a child file can clear e.g. `inference.extra_body`."""
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and value and isinstance(merged.get(key), dict):
            merged[key] = deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_raw(path: Path, seen: frozenset[Path]) -> dict[str, Any]:
    path = path.resolve()
    if path in seen:
        raise ValueError(f"circular 'extends' chain at {path}")
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"{path} must contain a mapping at the top level")
    parent = data.pop(EXTENDS_KEY, None)
    if parent is None:
        return data
    return deep_merge(_load_raw(path.parent / parent, seen | {path}), data)


def load_config(path: str | Path) -> AppConfig:
    return AppConfig(**_load_raw(Path(path), frozenset()))
