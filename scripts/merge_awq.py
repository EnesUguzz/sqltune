"""Merge a LoRA adapter into its bf16 base (optional) and quantize the result to AWQ 4-bit.

Purpose: A 7B model does not fit the 16 GB card in bf16 under vLLM. Merging the adapter and
quantizing with AWQ makes it servable. The base model is quantized with the same script and
the same calibration set, so the fine-tuned and the base 7B differ only by the fine-tune
(the official Qwen AWQ checkpoint uses another tool and generic calibration data).
Flow: bf16 base (CPU) [+ adapter -> merge_and_unload] -> llm-compressor AWQ scaling +
W4A16_ASYM quantization (group 128, lm_head kept in bf16; layers onloaded to the GPU one at a
time) calibrated on outputs/calib/<calib>.jsonl -> models/<name>-awq/ (compressed-tensors,
served by vLLM as `make serve MODEL=/models/<name>-awq`).
Params: --base, --adapter (omit to quantize the base alone), --calib, --out, --n (calibration
samples, typical 128), --max-len (tokens per sample, typical 1024; longer samples are skipped,
not truncated, so every sample keeps its SQL answer). Effect: one layer's cached activations
over all samples must fit in VRAM; on a 7B with 16 GB, 256 x 2048 (~190k tokens) spilled into
shared memory and ran ~300x slower, 128 x 1024 (~67k tokens) fits.
Caveat: the adapter was trained on the NF4 base and is folded into bf16 weights (standard
practice; the small mismatch is part of what the dev eval measures).
Runs in its own environment (llm-compressor pins its own torch / transformers):
`uv venv ~/venvs/awq && VIRTUAL_ENV=~/venvs/awq uv pip install llmcompressor peft`.
Try: `~/venvs/awq/bin/python scripts/merge_awq.py --adapter adapters/m1b_mix7b
--calib outputs/calib/m1b_mix7b.jsonl --out models/m1b_mix7b-awq`.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--base", default="Qwen/Qwen2.5-Coder-7B-Instruct")
    parser.add_argument("--adapter", type=Path, default=None)
    parser.add_argument("--calib", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--n", type=int, default=128)
    parser.add_argument("--max-len", type=int, default=1024)
    args = parser.parse_args()

    import torch
    from datasets import Dataset
    from llmcompressor import oneshot
    from llmcompressor.modifiers.quantization import QuantizationModifier
    from llmcompressor.modifiers.transform.awq import AWQModifier
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.base)
    model = AutoModelForCausalLM.from_pretrained(args.base, dtype=torch.bfloat16)
    if args.adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(args.adapter)).merge_and_unload()
        print(f"merged {args.adapter} into {args.base}")

    rows = [json.loads(line) for line in args.calib.read_text(encoding="utf-8").splitlines()]
    texts = [tokenizer.apply_chat_template(r["messages"], tokenize=False) for r in rows]
    texts = [
        t for t in texts if len(tokenizer(t, add_special_tokens=False).input_ids) <= args.max_len
    ]
    texts = texts[: args.n]
    print(f"{len(texts)} calibration samples of <= {args.max_len} tokens")
    ds = Dataset.from_dict({"text": texts}).map(
        lambda b: tokenizer(b["text"], add_special_tokens=False),
        batched=True,
        remove_columns=["text"],
    )
    recipe = [
        AWQModifier(),
        QuantizationModifier(ignore=["lm_head"], scheme="W4A16_ASYM", targets=["Linear"]),
    ]
    oneshot(
        model=model,
        dataset=ds,
        recipe=recipe,
        max_seq_length=args.max_len,
        num_calibration_samples=len(texts),
    )

    args.out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(args.out), save_compressed=True)
    tokenizer.save_pretrained(str(args.out))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
