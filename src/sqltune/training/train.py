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

import itertools
import json
import math
import os
import time
from pathlib import Path
from typing import Any

from sqltune.config.models import AppConfig, TrainConfig

# Tokens that open the assistant turn in Qwen's chat template; the loss mask must cover
# everything up to and including them.
ASSISTANT_HEADER = "<|im_start|>assistant\n"


def _sequences(batch: dict[str, Any]) -> list[tuple[list[int], list[int]]]:
    """(input_ids, labels) per example; padding-free batches pack several examples into one
    row, and their position_ids restart at 0 where each example begins."""
    rows = zip(batch["input_ids"].tolist(), batch["labels"].tolist(), strict=True)
    if "position_ids" not in batch:
        return list(rows)
    out = []
    for (ids, labels), pos in zip(rows, batch["position_ids"].tolist(), strict=True):
        starts = [i for i, p in enumerate(pos) if p == 0] + [len(pos)]
        out += [(ids[a:b], labels[a:b]) for a, b in itertools.pairwise(starts)]
    return out


def check_loss_mask(batch: dict[str, Any], tokenizer: Any, eos_id: int | None = None) -> None:
    """Fail fast if prompt tokens are not masked (-100), the completion is masked, or (with
    eos_id) the trained part lacks the end-of-turn token, so the model would never stop."""
    header = tokenizer.encode(ASSISTANT_HEADER, add_special_tokens=False)
    for ids, labels in _sequences(batch):
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
        if eos_id is not None and eos_id not in labels[prompt_end:]:
            raise AssertionError("trained completion does not end with the end-of-turn token")


def _fix_adapter_base(adapter_dir: Path, base_model: str) -> None:
    """Unsloth records its pre-quantized mirror (unsloth/...-bnb-4bit) as the base; vLLM
    serves the original model, so point the adapter back at it."""
    path = adapter_dir / "adapter_config.json"
    data = json.loads(path.read_text(encoding="utf-8"))
    data["base_model_name_or_path"] = base_model
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def _last_eval_loss(trainer: Any) -> float | None:
    losses = [h["eval_loss"] for h in trainer.state.log_history if "eval_loss" in h]
    return losses[-1] if losses else None


def warmup_steps(n_records: int, t: TrainConfig) -> int:
    """Warmup length in optimizer steps from TrainConfig.warmup_ratio (explicit, so it does not
    depend on how a transformers version interprets a fractional warmup_steps)."""
    per_epoch = math.ceil(
        n_records / (t.per_device_train_batch_size * t.gradient_accumulation_steps)
    )
    total = t.max_steps or math.ceil(per_epoch * t.num_train_epochs)
    return math.ceil(t.warmup_ratio * total)


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
    data = build_sft_dataset(cfg, tokenizer)
    dataset = data.train

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
        warmup_steps=warmup_steps(len(dataset), t),
        lr_scheduler_type=t.lr_scheduler_type,
        optim=t.optim,
        weight_decay=t.weight_decay,
        logging_steps=t.logging_steps,
        save_steps=t.save_steps,
        eval_strategy="steps" if data.val is not None else "no",
        eval_steps=t.eval_steps,
        per_device_eval_batch_size=t.per_device_train_batch_size,
        save_total_limit=2,
        seed=t.seed,
        bf16=True,
        report_to="mlflow",
    )
    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        train_dataset=dataset,
        eval_dataset=data.val,
        args=args,
    )
    check_loss_mask(
        next(iter(trainer.get_train_dataloader())),
        tokenizer,
        eos_id=tokenizer.convert_tokens_to_ids("<|im_end|>"),
    )

    mlflow.set_tracking_uri(cfg.evaluation.mlflow_tracking_uri)
    mlflow.set_experiment(cfg.run.experiment_name)
    os.environ["MLFLOW_TRACKING_URI"] = cfg.evaluation.mlflow_tracking_uri
    with mlflow.start_run(run_name=cfg.run.run_name) as run:  # the HF callback logs here
        torch.cuda.reset_peak_memory_stats()
        start = time.monotonic()
        result = trainer.train()
        runtime = time.monotonic() - start
        if data.val is not None:  # periodic evals stop before the last <eval_steps steps
            trainer.evaluate()

        model.save_pretrained(str(adapter_dir))
        tokenizer.save_pretrained(str(adapter_dir))
        _fix_adapter_base(adapter_dir, t.base_model)

        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total = sum(p.numel() for p in model.parameters())
        steps = trainer.state.global_step
        # tokens processed: one pass over the dataset per epoch (fractional with max_steps)
        train_ids: Any = trainer.train_dataset["input_ids"]  # type: ignore[index]
        tokens = sum(len(ids) for ids in train_ids) * (trainer.state.epoch or 0)
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
            "n_dropped_overlong": sum(data.dropped.values()),
            "final_eval_loss": _last_eval_loss(trainer),
            **{f"n_dropped_overlong_{k}": v for k, v in data.dropped.items()},
            **{f"n_rows_fallback_{k}": v for k, v in data.rows_fallback.items()},
            **{f"n_{part}_{k}": v[part] for k, v in data.counts.items() for part in v},
        }
        mlflow.log_metrics({k: float(v) for k, v in summary.items() if v is not None})
        record = {
            **summary,
            "mlflow_run_id": run.info.run_id,
            "dropped_by_db": data.dropped_by_db,
        }
        (results_dir / "train_summary.json").write_text(
            json.dumps(record, indent=2), encoding="utf-8"
        )
        mlflow.log_artifact(str(results_dir / "train_summary.json"))
    return adapter_dir
