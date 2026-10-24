"""Merge a LoRA adapter into its bf16 base and quantize the result to AWQ 4-bit for vLLM.

Purpose: A 7B model does not fit the 16 GB card in bf16 under vLLM. Merging the adapter and
quantizing with AWQ gives the same format as the official Qwen 7B AWQ baseline, so the
fine-tuned and the base 7B differ only by the fine-tune.
Flow: bf16 base (CPU) + adapters/<name> -> merge_and_unload -> llm-compressor AWQ
(W4A16_ASYM, group 128, lm_head kept in bf16; layers onloaded to the GPU one at a time)
calibrated on outputs/calib/<name>.jsonl -> models/<name>-awq/ (compressed-tensors, served by
vLLM as `make serve MODEL=/models/<name>-awq`).
Params: --base, --adapter, --calib, --out, --n (calibration samples, typical 256),
--max-len (tokens per sample, typical 2048).
Runs in its own environment (llm-compressor pins its own torch / transformers):
`uv venv ~/venvs/awq && VIRTUAL_ENV=~/venvs/awq uv pip install llmcompressor peft`.
Try: `~/venvs/awq/bin/python scripts/merge_awq.py --adapter adapters/m1b_mix7b`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", type=Path, required=True)
    parser.add_argument("--calib", type=Path, default=None)
    parser.add_argument("--out", type=Path, default=None)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--max-len", type=int, default=2048)
    args = parser.parse_args()
    name = args.adapter.name
    calib = args.calib or Path("outputs/calib") / f"{name}.jsonl"
    out = args.out or Path("models") / f"{name}-awq"

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from peft import PeftModel
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, torch_dtype=torch.bfloat16)
    model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
    print(f"merged {args.adapter} into {args.base}")

    rows = [json.loads(line) for line in calib.read_text(encoding="utf-8").splitlines()]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows[: args.n]]
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(
            b["text"], truncation=True, max_length=args.max_len, add_special_tokens=False
        ),
        batched=True,
        remove_columns=["text"],
    )
    recipe = [AWQModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"])]
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_len,
        num_calibration_samples=len(texts),
    )

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(out), save_compressed=True)
    tokenizer.save_pretrained(str(out))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
