"""Write a randomly initialised LoRA adapter, only to check that vLLM can load and serve one.

Purpose: vLLM's LoRA kernels are the piece most likely to break on a new GPU (Blackwell,
sm_120). Serving a dummy adapter on day one finds that out before any real training.
Flow: `sqltune dummy-adapter` -> make_dummy_adapter() -> adapters/dummy/ -> `make serve LORA=dummy`.
Params: r, lora_alpha, target_modules and base_model from TrainConfig (same shape as the
real adapter, so the check is representative).
Try: `make serve LORA=dummy` and send one chat request to model "sft"; the answer equals the
base model's, because peft initialises lora_B to zero (an untrained adapter is a no-op).
"""

from __future__ import annotations

from pathlib import Path

from sqltune.config.models import TrainConfig


def make_dummy_adapter(cfg: TrainConfig, out: Path) -> Path:
    """Save an untrained LoRA adapter for cfg.base_model into `out`; returns `out`."""
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForCausalLM

    model = AutoModelForCausalLM.from_pretrained(cfg.base_model, dtype=torch.bfloat16)
    lora = LoraConfig(
        r=cfg.r,
        lora_alpha=cfg.lora_alpha,
        lora_dropout=cfg.lora_dropout,
        target_modules=cfg.target_modules,
        task_type="CAUSAL_LM",
    )
    peft_model = get_peft_model(model, lora)
    out.mkdir(parents=True, exist_ok=True)
    peft_model.save_pretrained(str(out))
    return out
