from __future__ import annotations

import asyncio
import json
import random

import httpx

from sqltune.config.models import InferenceConfig
from sqltune.inference.openai_compat import OpenAICompatBackend

CFG = InferenceConfig(base_url="http://test/v1", model="base", concurrency=4, max_retries=2)


def _reply(content: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "x",
            "object": "chat.completion",
            "created": 0,
            "model": "base",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": content},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3, "total_tokens": 13},
        },
    )


def _client(handler: object) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))  # type: ignore[arg-type]


def test_order_preserved_under_concurrency() -> None:
    async def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        await asyncio.sleep(random.uniform(0, 0.05))
        return _reply(body["messages"][-1]["content"].upper())

    backend = OpenAICompatBackend(CFG, http_client=_client(handler))
    reqs = [[{"role": "user", "content": f"q{i}"}] for i in range(20)]
    out = backend.complete_many(reqs)
    assert [c.text for c in out] == [f"Q{i}" for i in range(20)]
    assert out[0].prompt_tokens == 10
    assert out[0].completion_tokens == 3


def test_request_carries_settings() -> None:
    seen: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append(json.loads(request.content))
        return _reply("ok")

    cfg = CFG.model_copy(update={"extra_body": {"top_k": 1}, "max_tokens": 77})
    OpenAICompatBackend(cfg, http_client=_client(handler)).complete_many(
        [[{"role": "user", "content": "q"}]]
    )
    assert seen[0]["model"] == "base"
    assert seen[0]["temperature"] == 0.0
    assert seen[0]["max_tokens"] == 77
    assert seen[0]["top_k"] == 1


def test_retry_then_success() -> None:
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] == 1:
            return httpx.Response(503, json={"error": "busy"})
        return _reply("SELECT 1")

    out = OpenAICompatBackend(CFG, http_client=_client(handler)).complete_many(
        [[{"role": "user", "content": "q"}]]
    )
    assert out[0].text == "SELECT 1"
    assert out[0].error is None
    assert calls["n"] == 2


def test_permanent_failure_is_error() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(500, json={"error": "boom"})

    out = OpenAICompatBackend(CFG, http_client=_client(handler)).complete_many(
        [[{"role": "user", "content": "q"}]]
    )
    assert out[0].text == ""
    assert out[0].error is not None
