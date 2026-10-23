"""Send many chat requests to an OpenAI-compatible server (vLLM) concurrently.

Purpose: vLLM batches concurrent requests on the GPU (continuous batching), so 1034
questions take minutes when sent in parallel instead of an hour one by one.
Flow: runner -> complete_many(list of message lists) -> Completion per request, same order.
Params: InferenceConfig (model, base_url, temperature, max_tokens, extra_body, concurrency,
timeout_s, max_retries).
Try: `make serve`, then
`OpenAICompatBackend(InferenceConfig(model="base")).complete_many([[{"role": "user", "content": "hi"}]])`.
"""

from __future__ import annotations

import asyncio
import time

import httpx
from openai import APIConnectionError, APIStatusError, APITimeoutError, AsyncOpenAI
from pydantic import BaseModel
from tenacity import (
    AsyncRetrying,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from sqltune.config.models import InferenceConfig

RETRYABLE = (APIConnectionError, APITimeoutError, APIStatusError)


class Completion(BaseModel):
    text: str
    latency_ms: float
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    error: str | None = None


class OpenAICompatBackend:
    def __init__(self, cfg: InferenceConfig, http_client: httpx.AsyncClient | None = None) -> None:
        self.cfg = cfg
        self._http_client = http_client

    async def _one(
        self, client: AsyncOpenAI, sem: asyncio.Semaphore, messages: list[dict[str, str]]
    ) -> Completion:
        async with sem:
            start = time.monotonic()
            try:
                async for attempt in AsyncRetrying(
                    retry=retry_if_exception_type(RETRYABLE),
                    stop=stop_after_attempt(self.cfg.max_retries + 1),
                    wait=wait_exponential(multiplier=0.5, max=8),
                    reraise=True,
                ):
                    with attempt:
                        resp = await client.chat.completions.create(
                            model=self.cfg.model,
                            messages=messages,  # type: ignore[arg-type]
                            temperature=self.cfg.temperature,
                            max_tokens=self.cfg.max_tokens,
                            extra_body=self.cfg.extra_body or None,
                        )
            except Exception as exc:
                return Completion(
                    text="", latency_ms=(time.monotonic() - start) * 1000, error=repr(exc)[:500]
                )
            usage = resp.usage
            return Completion(
                text=resp.choices[0].message.content or "",
                latency_ms=(time.monotonic() - start) * 1000,
                prompt_tokens=usage.prompt_tokens if usage else None,
                completion_tokens=usage.completion_tokens if usage else None,
            )

    async def _many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        client = AsyncOpenAI(
            base_url=self.cfg.base_url,
            api_key="EMPTY",
            timeout=self.cfg.timeout_s,
            max_retries=0,  # retries are tenacity's job, so they are counted in one place
            # None = the SDK builds its own client; tests inject an httpx mock transport
            http_client=self._http_client,  # type: ignore[arg-type]  # SDK annotates httpx2
        )
        sem = asyncio.Semaphore(self.cfg.concurrency)
        try:
            return list(await asyncio.gather(*(self._one(client, sem, m) for m in requests)))
        finally:
            await client.close()

    def complete_many(self, requests: list[list[dict[str, str]]]) -> list[Completion]:
        """One Completion per request, in input order; failures carry `error`, never raise."""
        return asyncio.run(self._many(requests))
