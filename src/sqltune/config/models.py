"""Typed configuration for every tunable value in sqltune.

Purpose: Single source of truth for parameters. Every field documents what it does, its
typical range and what happens to the metrics when it changes, so experiments can be
reasoned about without reading the implementation.
Flow: experiments/configs/*.yaml -> config/loader.py -> AppConfig -> every component.
Params: all of them; each field's `description` is the documentation.
Try: Copy experiments/configs/m0/baseline.yaml, change one field, `make eval CONFIG=<file>`.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class _Section(BaseModel):
    model_config = ConfigDict(extra="forbid")


class DataConfig(_Section):
    data_dir: Path = Field(
        Path("data"),
        description=(
            "What: root folder for downloaded datasets (git-ignored). Typical: data. "
            "Effect: none on metrics; Spider lands in <data_dir>/spider/."
        ),
    )
    split: Literal["train", "dev"] = Field(
        "dev",
        description=(
            "What: Spider split to evaluate on. Typical: dev. "
            "Effect: 'train' only for sanity checks; scores on train are inflated after SFT."
        ),
    )
    subset: Literal["full", "dev_small"] = Field(
        "full",
        description=(
            "What: 'full' = every question of the split, 'dev_small' = a fixed random sample "
            "of dev_small_size questions. Typical: full (1034 dev questions take minutes on vLLM). "
            "Effect: dev_small is only for smoke tests; small samples make EX differences noisy "
            "(300 questions ~ +/-2.7 points standard error)."
        ),
    )
    dev_small_size: int = Field(
        300,
        ge=1,
        description=(
            "What: number of questions in the dev_small subset. Typical: 30 for smoke tests. "
            "Effect: larger = less noise, slower."
        ),
    )
    seed: int = Field(
        13,
        description=(
            "What: random seed for drawing dev_small. Typical: any fixed integer. "
            "Effect: a different seed draws a different subset; keep it fixed so runs compare."
        ),
    )
    spider_drive_id: str = Field(
        "1403EGqzIDoHMdQF4c9Bkyl7dZLZ5Wt6J",
        description=(
            "What: Google Drive file id of the official Spider zip (questions + SQLite DBs). "
            "Typical: the id from the Spider site. Effect: none on metrics; change it only if "
            "the official link moves."
        ),
    )


class PromptConfig(_Section):
    system_template: Path = Field(
        Path("prompts/sql_system.md"),
        description=(
            "What: system message file shared by every model and by SFT. "
            "Effect: changing it changes every row of the comparison table; baseline and "
            "fine-tune must use the same file or the delta is meaningless."
        ),
    )
    single_template: Path = Field(
        Path("prompts/sql_single.md"),
        description=(
            "What: user message template with {schema} and {question} placeholders. "
            "Effect: same as system_template; SFT trains on exactly this text."
        ),
    )


class InferenceConfig(_Section):
    model: str = Field(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        description=(
            "What: model name sent to the OpenAI-compatible server. With the repo's vLLM "
            "compose this is 'base' (served base model) or 'sft' (LoRA adapter). "
            "Effect: picks which model is evaluated."
        ),
    )
    base_url: str = Field(
        "http://localhost:8000/v1",
        description=(
            "What: OpenAI-compatible endpoint. Typical: http://localhost:8000/v1 (vLLM). "
            "Effect: none on metrics."
        ),
    )
    temperature: float = Field(
        0.0,
        ge=0.0,
        le=2.0,
        description=(
            "What: sampling temperature. Typical: 0.0 for evaluation. "
            "Effect: >0 makes runs non-deterministic and usually lowers EX."
        ),
    )
    max_tokens: int = Field(
        512,
        ge=16,
        description=(
            "What: max generated tokens per answer. Typical: 512 (Spider SQL is < 200 tokens). "
            "Effect: too low truncates SQL (shows up as no_sql / exec_error)."
        ),
    )
    extra_body: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "What: extra JSON fields passed to the server as-is (e.g. vLLM sampling options). "
            "Typical: {}. Effect: depends on the field; recorded in the run config."
        ),
    )
    concurrency: int = Field(
        16,
        ge=1,
        description=(
            "What: max in-flight requests. Typical: 8-32 for vLLM on one GPU. "
            "Effect: higher = faster runs until the GPU saturates; no effect on accuracy."
        ),
    )
    timeout_s: float = Field(
        120.0,
        gt=0,
        description=(
            "What: per-request HTTP timeout in seconds. Typical: 60-120. "
            "Effect: timed-out requests are retried, then recorded as errors."
        ),
    )
    max_retries: int = Field(
        3,
        ge=0,
        description=(
            "What: retries per request on transient errors (exponential backoff). "
            "Typical: 3. Effect: none on accuracy; fewer recorded errors."
        ),
    )


class EvalConfig(_Section):
    exec_timeout_s: float = Field(
        5.0,
        gt=0,
        description=(
            "What: max seconds one SQL query may run against SQLite. Typical: 5. "
            "Effect: runaway queries (cartesian products, recursive CTEs) become 'timeout' "
            "instead of hanging the run; too low marks slow but correct queries wrong."
        ),
    )
    max_result_rows: int = Field(
        100_000,
        ge=1,
        description=(
            "What: max rows fetched from one query. Typical: 100000. "
            "Effect: larger results become 'too_many_rows' (protects memory)."
        ),
    )
    exec_workers: int = Field(
        8,
        ge=1,
        description=(
            "What: parallel threads executing SQL for scoring. Typical: CPU cores. "
            "Effect: speed only."
        ),
    )
    mlflow_tracking_uri: str = Field(
        "sqlite:///mlflow.db",
        description=(
            "What: MLflow tracking store. Typical: sqlite:///mlflow.db (local). "
            "Effect: none on metrics; `mlflow ui --backend-store-uri sqlite:///mlflow.db`."
        ),
    )


class TrainConfig(_Section):
    base_model: str = Field(
        "Qwen/Qwen2.5-Coder-3B-Instruct",
        description=(
            "What: Hugging Face id of the model to fine-tune. Typical: the evaluated base. "
            "Effect: must match the model vLLM serves, or the adapter will not load."
        ),
    )
    load_in_4bit: bool = Field(
        True,
        description=(
            "What: True = QLoRA (frozen base in 4-bit NF4), False = LoRA (frozen base in bf16). "
            "Typical: True. Effect: 4-bit cuts training VRAM ~3x; the adapter is 16-bit either "
            "way. Small quality cost possible; M1 compares both."
        ),
    )
    max_seq_length: int = Field(
        2048,
        ge=256,
        description=(
            "What: max tokens per training example (prompt + SQL). Typical: 2048 for Spider. "
            "Effect: longer examples are dropped and counted; higher = more VRAM."
        ),
    )
    r: int = Field(
        16,
        ge=1,
        description=(
            "What: LoRA rank (width of the low-rank update). Typical: 8-64. "
            "Effect: higher = more capacity and trainable params, more VRAM, more overfit risk."
        ),
    )
    lora_alpha: int = Field(
        16,
        ge=1,
        description=(
            "What: LoRA scaling numerator; update is scaled by alpha / r. Typical: = r or 2r. "
            "Effect: higher = larger effective step for the adapter (interacts with lr)."
        ),
    )
    lora_dropout: float = Field(
        0.0,
        ge=0.0,
        le=0.5,
        description=(
            "What: dropout on the LoRA input. Typical: 0.0 (Unsloth fast path) to 0.1. "
            "Effect: >0 regularizes a bit but disables some Unsloth optimizations."
        ),
    )
    target_modules: list[str] = Field(
        default_factory=lambda: [
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
        ],
        description=(
            "What: linear layers that get LoRA adapters. Typical: all attention + MLP projections. "
            "Effect: fewer modules = fewer params, usually lower quality."
        ),
    )
    learning_rate: float = Field(
        2e-4,
        gt=0,
        description=(
            "What: peak learning rate. Typical: 1e-4 to 3e-4 for LoRA. "
            "Effect: too high = unstable loss / forgetting; too low = slow, under-fit."
        ),
    )
    num_train_epochs: float = Field(
        2.0,
        gt=0,
        description=(
            "What: passes over Spider train. Typical: 1-3. "
            "Effect: more epochs fit Spider's style better and risk overfitting to it."
        ),
    )
    max_steps: int | None = Field(
        None,
        ge=1,
        description=(
            "What: hard cap on optimizer steps (overrides epochs when set). "
            "Typical: null; 50 for smoke tests. Effect: short runs only check the pipeline."
        ),
    )
    per_device_train_batch_size: int = Field(
        8,
        ge=1,
        description=(
            "What: examples per forward pass. Typical: 4-16 on 16 GB with a 3B QLoRA. "
            "Effect: higher = faster, more VRAM; lower it on OOM and raise grad accumulation."
        ),
    )
    gradient_accumulation_steps: int = Field(
        2,
        ge=1,
        description=(
            "What: forward passes per optimizer step. Effective batch = batch x this. "
            "Typical: keep effective batch ~16. Effect: same math as a bigger batch, slower."
        ),
    )
    warmup_ratio: float = Field(
        0.03,
        ge=0.0,
        le=0.5,
        description=(
            "What: fraction of steps with linearly rising lr. Typical: 0.03-0.1. "
            "Effect: avoids early instability."
        ),
    )
    lr_scheduler_type: str = Field(
        "cosine",
        description=(
            "What: lr schedule after warmup (transformers names). Typical: cosine or linear. "
            "Effect: small; cosine ends with a gentler lr."
        ),
    )
    weight_decay: float = Field(
        0.0,
        ge=0.0,
        description=(
            "What: AdamW weight decay on adapter weights. Typical: 0.0-0.01. "
            "Effect: mild regularization."
        ),
    )
    logging_steps: int = Field(
        10,
        ge=1,
        description="What: log loss every N steps. Typical: 10. Effect: logging granularity only.",
    )
    save_steps: int = Field(
        200,
        ge=1,
        description=(
            "What: checkpoint every N steps. Typical: 200. "
            "Effect: disk use; lets a crashed run resume."
        ),
    )
    seed: int = Field(
        13,
        description=(
            "What: training seed (adapter init, data order). Typical: fixed integer. "
            "Effect: different seeds give ~+/-1 point EX variation."
        ),
    )
    output_dir: Path = Field(
        Path("adapters"),
        description=(
            "What: folder where adapters are saved as <output_dir>/<run_name>/ (git-ignored). "
            "Effect: vLLM mounts this folder to serve the adapter."
        ),
    )


class RunConfig(_Section):
    experiment_name: str = Field(
        "m0",
        description="What: MLflow experiment name. Typical: the milestone (m0, m1). Effect: grouping only.",
    )
    run_name: str = Field(
        ...,
        description=(
            "What: unique name of this run; results go to <results_dir>/<run_name>/. "
            "Effect: reusing a name overwrites that run's results."
        ),
    )
    reference_run: str | None = Field(
        None,
        description=(
            "What: earlier run to compare against in the report (per-metric delta, "
            "fixed/broken examples, McNemar p). Typical: m0_baseline. Effect: report only."
        ),
    )
    results_dir: Path = Field(
        Path("results"),
        description="What: folder for run outputs (committed). Effect: none on metrics.",
    )


class AppConfig(BaseSettings):
    model_config = SettingsConfigDict(
        extra="forbid", env_prefix="SQLTUNE_", env_nested_delimiter="__"
    )

    data: DataConfig = Field(default_factory=DataConfig, description="What: dataset settings.")
    prompt: PromptConfig = Field(default_factory=PromptConfig, description="What: prompt files.")
    inference: InferenceConfig = Field(
        default_factory=InferenceConfig, description="What: model server settings."
    )
    evaluation: EvalConfig = Field(default_factory=EvalConfig, description="What: SQL scoring.")
    training: TrainConfig = Field(default_factory=TrainConfig, description="What: QLoRA/LoRA SFT.")
    run: RunConfig = Field(..., description="What: run naming and outputs.")
