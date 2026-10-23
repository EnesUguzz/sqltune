"""Check that vLLM tokenizes eval prompts exactly like the trainer's (Unsloth-saved) tokenizer.

Purpose: Train/serve prompt drift (chat template, add_generation_prompt) silently lowers a
fine-tune's score; this compares token ids for a sample of dev prompts on both served models.
Flow: adapters/<adapter>/ tokenizer (saved by the trainer) vs POST /tokenize on vLLM.
Params: --adapter (folder under adapters/), --every (take every Nth dev question).
Try: `make serve LORA=m0_qlora && uv run --extra train python scripts/check_tokenize.py`.
"""

from __future__ import annotations

import argparse
import sys

import httpx
from transformers import AutoTokenizer

from sqltune.config.models import DataConfig, PromptConfig
from sqltune.data.spider import load_spider
from sqltune.prompting.builder import build_messages


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--adapter", default="m0_qlora")
    parser.add_argument("--every", type=int, default=100)
    parser.add_argument("--url", default="http://localhost:8000/tokenize")
    args = parser.parse_args()

    tok = AutoTokenizer.from_pretrained(f"adapters/{args.adapter}")
    sample = load_spider(DataConfig(), "dev")[:: args.every]
    checks = same = 0
    for ex in sample:
        msgs = build_messages(ex, PromptConfig())
        ids = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True)
        ids = ids["input_ids"] if hasattr(ids, "keys") else ids
        for model in ("base", "sft"):
            body = {"model": model, "messages": msgs, "add_generation_prompt": True}
            served = httpx.post(args.url, json=body, timeout=30).json()["tokens"]
            checks += 1
            same += served == ids
    print(f"identical prompt tokens: {same}/{checks}")
    return 0 if same == checks else 1


if __name__ == "__main__":
    sys.exit(main())
