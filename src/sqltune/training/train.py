"""QLoRA / LoRA supervised fine-tuning with Unsloth + TRL, logged to MLflow.

Purpose: Teach the base model Spider's question -> SQL mapping with a small trainable adapter
while the base stays frozen (4-bit for QLoRA), so a 3B model trains on one 16 GB GPU.
Flow: TrainConfig -> FastLanguageModel (4-bit base + LoRA) -> build_sft_dataset ->
TRL SFTTrainer (loss on the SQL completion only) -> adapters/<run_name>/ +
results/<run_name>/train_summary.json -> MLflow (loss curve + summary).
Params: every TrainConfig field; the prompt comes from PromptConfig (same as evaluation).
Try: `make train CONFIG=experiments/configs/m0/smoke.yaml` (50 steps, a few minutes).
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

from sqltune.config.models import AppConfig

# Tokens that open the assistant turn in Qwen's chat template; the loss mask must cover
# everything up to and including them.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def check_loss_mask(batch: dict[str, Any], tokenizer: Any) -> None:
    """Fail fast if prompt tokens are not masked (-100) or the completion is masked."""
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
    for ids, labels in zip(batch["input_ids"].tolist(), batch["labels"].tolist(), strict=True):
        starts = [
            i for i in range(len(ids) - len(header) + 1) if ids[i : i + len(header)] == header
        ]
        if not starts:
            raise AssertionError("assistant header not found in a training example")
        prompt_end = starts[-1] + len(header)
        if any(label != -100 for label in labels[:prompt_end]):
            raise AssertionError("prompt tokens contribute to the loss (completion_only_loss off?)")
        if labels[prompt_end] == -100:
            raise AssertionError("first completion token is masked; nothing would be learned")


def _fix_adapter_base(adapter_dir: Path, base_model: str) -> None:
    """Unsloth records its pre-quantized mirror (unsloth/...-bnb-4bit) as the base; vLLM
    serves the original model, so point the adapter back at it."""
    path = adapter_dir / "adapter_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["base_model_name_or_path"] = base_model
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def train(cfg: AppConfig) -> Path:
    import mlflow
    import torch
    import unsloth  # noqa: F401  # must be imported before trl / transformers to patch them
    from trl import SFTConfig, SFTTrainer  # type: ignore[attr-defined]  # lazy exports; Unsloth patches these names
    from unsloth import FastLanguageModel

    from sqltune.training.sft_data import build_sft_dataset

    t = cfg.training
    adapter_dir = t.output_dir / cfg.run.run_name
    results_dir = cfg.run.results_dir / cfg.run.run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=t.base_model,
        max_seq_length=t.max_seq_length,
        load_in_4bit=t.load_in_4bit,
        dtype=None,
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=t.r,
        lora_alpha=t.lora_alpha,
        lora_dropout=t.lora_dropout,
        target_modules=t.target_modules,
        bias="none",
        use_gradient_checkpointing="unsloth",
        random_state=t.seed,
    )
    dataset, n_dropped = build_sft_dataset(cfg, tokenizer)

    args = SFTConfig(
        output_dir=str(adapter_dir / "checkpoints"),
        run_name=cfg.run.run_name,
        max_length=t.max_seq_length,
        completion_only_loss=True,
        packing=False,
        per_device_train_batch_size=t.per_device_train_batch_size,
        gradient_accumulation_steps=t.gradient_accumulation_steps,
        learning_rate=t.learning_rate,
        num_train_epochs=t.num_train_epochs,
        max_steps=t.max_steps or -1,
        warmup_steps=t.warmup_ratio,  # transformers 5: a value < 1 is a ratio of total steps
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        weight_decay=t.weight_decay,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        save_total_limit=2,
        seed=t.seed,
        bf16=True,
        report_to="mlflow",
    )
    trainer = SFTTrainer(model=model, processing_class=tokenizer, train_dataset=dataset, args=args)
    check_loss_mask(next(iter(trainer.get_train_dataloader())), tokenizer)

    mlflow.set_tracking_uri(cfg.evaluation.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run.experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = cfg.evaluation.mlflow_tracking_uri
    with mlflow.start_run(run_name=cfg.run.run_name):  # the HF callback logs into this run
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        result = trainer.train()
        runtime = time.monotonic() - start

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        _fix_adapter_base(adapter_dir, t.base_model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        steps = trainer.state.global_step
        tokens = next(
            (h["num_tokens"] for h in reversed(trainer.state.log_history) if "num_tokens" in h),
            None,
        )
        summary = {
            "runtime_s": runtime,
            "steps": steps,
            "steps_per_s": steps / runtime,
            "tokens_per_s": tokens / runtime if tokens else None,
            "peak_vram_gib": torch.cuda.max_memory_allocated() / 2**30,
            "final_train_loss": result.training_loss,
            "trainable_params": trainable,
            "trainable_ratio": trainable / total,
            "n_records": len(dataset),
            "n_dropped_overlong": n_dropped,
        }
        (results_dir / "train_summary.json").write_text(
            json.dumps(summary, indent=2), encoding="utf-8"
        )
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if v is not None})
        mlflow.log_artifact(str(results_dir / "train_summary.json"))
    return adapter_dir
