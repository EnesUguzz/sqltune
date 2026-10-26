"""Ask Claude (e.g. Haiku) through the local `claude -p` CLI, as a closed-model reference row.

Purpose: Compare the fine-tuned open models with a closed API model on the same prompt,
without an API key: `claude -p` runs on the user's Claude subscription. Our system prompt
replaces Claude Code's own (--system-prompt), and tools, MCP servers and settings are off,
so the model answers one-shot like every other row.
Flow: runner -> complete_many([system, user] messages) -> one `claude -p` subprocess per
request (thread pool) -> JSON result -> Completion, same order.
Params: InferenceConfig.model (CLI model alias, e.g. "haiku"), concurrency, timeout_s,
max_retries, cli_command. Only single-turn requests (system + user) are supported, so
few-shot and retry variants are refused.
Try: `ClaudeCLIBackend(InferenceConfig(backend="claude_cli", model="haiku")).complete_many(
[[{"role": "system", "content": "Be brief."}, {"role": "user", "content": "SELECT 1?"}]])`.
"""

from __future__ import annotations

import json
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor

from sqltune.config.models import InferenceConfig
from sqltune.inference.openai_compat import Completion


def cli_args(cfg: InferenceConfig, system: str) -> list[str]:
    return [
        *cfg.cli_command,
        "-p",
        "--model",
        cfg.model,
        "--system-prompt",
        system,
        "--tools",
        "",
        "--strict-mcp-config",
        "--setting-sources",
        "",
        "--no-session-persistence",
        "--output-format",
        "json",
    ]


def split_messages(messages: list[dict[str, str]]) -> tuple[str, str]:
    roles = [m["role"] for m in messages]
    if roles != ["system", "user"]:
        raise ValueError(f"claude_cli backend takes [system, user] messages only, got {roles}")
    return messages[0]["content"], messages[1]["content"]


class ClaudeCLIBackend:
    def __init__(self, cfg: InferenceConfig) -> None:
        self.cfg = cfg

    def _one(self, messages: list[dict[str, str]]) -> Completion:
        system, user = split_messages(messages)
        start = time.monotonic()
        error = ""
        for _ in range(self.cfg.max_retries + 1):
            try:
                proc = subprocess.run(
                    cli_args(self.cfg, system),
                    input=user,
                    capture_output=True,
                    text=True,
                    encoding="utf-8",
                    timeout=self.cfg.timeout_s,
                    check=False,
                )
                data = json.loads(proc.stdout)
                if data.get("is_error") or "result" not in data:
                    error = str(data)[:500]
                    continue
                usage = data.get("usage") or {}
                return Completion(
                    text=data["result"],
                    latency_ms=(time.monotonic() - start) * 1000,
                    prompt_tokens=sum(
                        usage.get(k, 0)
                        for k in (
                            "input_tokens",
                            "cache_creation_input_tokens",
                            "cache_read_input_tokens",
                        )
                    ),
                    completion_tokens=usage.get("output_tokens"),
                )
            except (subprocess.TimeoutExpired, json.JSONDecodeError, OSError) as exc:
                error = repr(exc)[:500]
        return Completion(text="", latency_ms=(time.monotonic() - start) * 1000, error=error)

    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        """One Completion per request, in input order; failures carry `error`, never raise."""
        for messages in requests:
            split_messages(messages)  # refuse multi-turn variants before spending any calls
        with ThreadPoolExecutor(max_workers=self.cfg.concurrency) as pool:
            return list(pool.map(self._one, requests))
