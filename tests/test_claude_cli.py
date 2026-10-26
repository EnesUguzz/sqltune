from __future__ import annotations

import json
import subprocess
from typing import Any

import pytest

from sqltune.config.models import InferenceConfig
from sqltune.inference import claude_cli
from sqltune.inference.claude_cli import ClaudeCLIBackend, cli_args

MESSAGES = [{"role": "system", "content": "SYS"}, {"role": "user", "content": "Q?"}]


def test_cli_args_replace_system_prompt_and_disable_tools() -> None:
    args = cli_args(InferenceConfig(backend="claude_cli", model="haiku"), "SYS")
    assert args[:2] == ["claude", "-p"]
    assert args[args.index("--model") + 1] == "haiku"
    assert args[args.index("--system-prompt") + 1] == "SYS"
    assert args[args.index("--tools") + 1] == ""
    assert "--strict-mcp-config" in args


def test_result_and_usage_are_parsed(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}

    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        seen["input"] = kw["input"]
        out = {"result": "```sql\nSELECT 1\n```", "usage": {"input_tokens": 5, "output_tokens": 7}}
        return subprocess.CompletedProcess(args, 0, stdout=json.dumps(out), stderr="")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    [c] = ClaudeCLIBackend(InferenceConfig(backend="claude_cli")).complete_many([MESSAGES])
    assert seen["input"] == "Q?"
    assert c.text == "```sql\nSELECT 1\n```"
    assert (c.prompt_tokens, c.completion_tokens, c.error) == (5, 7, None)


def test_errors_are_recorded_not_raised(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_run(args: list[str], **kw: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args, 1, stdout="not json", stderr="boom")

    monkeypatch.setattr(claude_cli.subprocess, "run", fake_run)
    cfg = InferenceConfig(backend="claude_cli", max_retries=1)
    [c] = ClaudeCLIBackend(cfg).complete_many([MESSAGES])
    assert c.text == "" and c.error is not None


def test_multi_turn_requests_are_refused() -> None:
    retry = [*MESSAGES, {"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
    with pytest.raises(ValueError, match="system, user"):
        ClaudeCLIBackend(InferenceConfig(backend="claude_cli")).complete_many([retry])
