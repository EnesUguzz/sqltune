"""Completion-only loss on the M1b validation split for a base model and base + adapter.

Purpose: Shows how much a fine-tune learned on databases it never saw, measured the same way
for the base model (which the Trainer's periodic evaluation never scores) and the adapter.
Flow: train config -> Unsloth 4-bit base (as in training) -> build_sft_dataset().val ->
per record: loss on the SQL completion tokens only (logits kept for the completion only) ->
token-weighted mean, base first, then with the adapter loaded.
Params: positional: train config, adapter dir. The mean is over all completion tokens, so it
differs slightly from the Trainer's per-batch mean.
Try: `uv run --extra train python scripts/val_loss.py experiments/configs/m1b/train/mix.yaml
adapters/m1b_mix` (GPU, with vLLM stopped).
"""

from __future__ import annotations

import sys
from typing import Any


def main() -> None:
    import unsloth  # noqa: F401, I001  # must be imported before transformers / peft
    import torch
    from peft import PeftModel
    from unsloth import FastLanguageModel

    from sqltune.config.loader import load_config
    from sqltune.training.sft_data import build_sft_dataset

    cfg_path, adapter = sys.argv[1], sys.argv[2]
    t = load_config(cfg_path).training
    model, tok = FastLanguageModel.from_pretrained(
        model_name=t.base_model, max_seq_length=t.max_seq_length, load_in_4bit=True, dtype=None
    )
    val = build_sft_dataset(load_config(cfg_path), tok).val
    assert val is not None, "the config has no validation split (val_fraction 0)"

    def ids_of(messages: list[dict[str, str]], gen: bool = False) -> list[int]:
        out: Any = tok.apply_chat_template(messages, tokenize=True, add_generation_prompt=gen)
        return list(out["input_ids"] if hasattr(out, "keys") else out)

    def val_loss(m: Any) -> tuple[float, int]:
        m.eval()
        total, n_tok = 0.0, 0
        with torch.no_grad():
            for r in val:
                p = ids_of(r["prompt"], gen=True)
                full = ids_of(r["prompt"] + r["completion"])
                assert full[: len(p)] == p, "prompt is not a prefix of prompt + completion"
                k = len(full) - len(p)  # completion tokens; predicted from positions p-1 .. -2
                out = m(input_ids=torch.tensor([full], device="cuda"), logits_to_keep=k + 1)
                pred = out.logits[0, :-1].float()
                target = torch.tensor(full[len(p) :], device="cuda")
                total += torch.nn.functional.cross_entropy(pred, target, reduction="sum").item()
                n_tok += k
        return total / n_tok, n_tok

    FastLanguageModel.for_inference(model)
    base, n = val_loss(model)
    print(f"{t.base_model}: base val loss {base:.4f} over {n} SQL tokens ({len(val)} records)")
    model = PeftModel.from_pretrained(model, adapter)
    tuned, _ = val_loss(model)
    print(f"{t.base_model}: + {adapter} val loss {tuned:.4f}")


if __name__ == "__main__":
    main()
