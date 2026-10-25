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
    split: Literal["train", "dev", "test"] = Field(
        "dev",
        description=(
            "What: Spider split to evaluate on. Typical: dev. "
            "Effect: 'train' only for sanity checks; scores on train are inflated after SFT. "
            "'test' is run once per final model (needs run.test_once)."
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
    dataset: Literal["spider", "bird"] = Field(
        "spider",
        description=(
            "What: benchmark to evaluate on. 'bird' = BIRD Mini-Dev (SQLite, 500 questions, "
            "with evidence). Effect: picks loader and databases; training always uses Spider."
        ),
    )
    train_sources: list[Literal["spider", "bird"]] = Field(
        ["spider"],
        min_length=1,
        description=(
            "What: datasets the fine-tune is trained on (Spider train_spider.json, BIRD "
            "train.json). Typical: [spider] (M0/M1a), [spider, bird] (M1b). Effect: adding BIRD "
            "brings evidence hints and large real-world schemas into training."
        ),
    )
    spider_train_exclude_dbs: list[str] = Field(
        [],
        description=(
            "What: Spider train databases left out of training. Typical: [] for Spider-only "
            "runs (keeps M0/M1a reproducible); [formula_1, soccer_1] whenever BIRD is "
            "evaluated: formula_1 is the same data as BIRD Mini-Dev formula_1 (bird_diag) and "
            "soccer_1 shares 6 of 7 tables with european_football_2 (bird_holdout). "
            "Effect: 80 + 14 questions fewer; no BIRD eval database is seen in training."
        ),
    )
    bird_train_exclude_dbs: list[str] = Field(
        ["world", "airline"],
        description=(
            "What: BIRD train databases left out of training because they duplicate an eval "
            "database (table-name fingerprint, tests/test_bird_train.py): `world` holds the "
            "same data as Spider dev `world_1`; `airline` shares its table names with Spider "
            "dev `flight_2`. Typical: leave as is. Effect: 191 questions fewer; no eval "
            "database is seen in training."
        ),
    )
    bird_train_skip_failing_gold: bool = Field(
        True,
        description=(
            "What: skip BIRD train questions listed in bird_train/gold_failures.json (gold SQL "
            "that errors, times out or returns too many rows; written by "
            "`make dataset-review`; loading fails if the file is missing). Typical: true. Effect: the model is not trained to "
            "reproduce SQL that does not run."
        ),
    )
    bird_diag_dbs: list[str] = Field(
        [
            "california_schools",
            "card_games",
            "formula_1",
            "superhero",
            "thrombosis_prediction",
        ],
        description=(
            "What: the BIRD Mini-Dev databases that form the diag half; every other database "
            "is holdout. Chosen once as the subset whose question count is closest to 250 of "
            "500 and pinned here so no other setting (e.g. `seed`) can move holdout questions "
            "into diag. Typical: leave as is. Effect: changing it changes which questions you "
            "tune on; never put a holdout database here."
        ),
    )
    bird_split: Literal["diag", "holdout", "all"] = Field(
        "diag",
        description=(
            "What: half of BIRD Mini-Dev, split by database. 'diag' is used for decisions and "
            "diagnostics; 'holdout' is the headline set, run once per final model. "
            "'all' includes holdout, so it also needs run.test_once. "
            "Effect: keeps the headline number free of tuning."
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
    include_train_others: bool = Field(
        False,
        description=(
            "What: also train on Spider's train_others.json (1659 questions on 6 more "
            "databases). Typical: false; true for the data-diversity experiment. "
            "Effect: more schemas seen in training; ~24% more training time."
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

    evidence_template: Path = Field(
        Path("prompts/sql_single_evidence.md"),
        description=(
            "What: user template used when the example has evidence (BIRD's hint field), with "
            "{schema}, {evidence} and {question}. Effect: BIRD's standard setup; Spider never "
            "uses it, so Spider prompts are unchanged."
        ),
    )
    retry_template: Path = Field(
        Path("prompts/sql_retry.md"),
        description=(
            "What: follow-up message after a query fails, with {error} (the SQLite error text "
            "only). Effect: used only when inference.retry_on_error is true."
        ),
    )
    sample_rows: int = Field(
        0,
        ge=0,
        le=10,
        description=(
            "What: example rows shown under each CREATE TABLE. Typical: 0 or 3. Effect: the "
            "model sees real values and spellings (helps with invented columns / values); "
            "longer prompts, slower; a fine-tune trained without rows sees a new prompt."
        ),
    )
    few_shot_k: int = Field(
        0,
        ge=0,
        le=8,
        description=(
            "What: solved Spider-train examples (from other databases) shown before the "
            "question. Typical: 0 or 3. Effect: teaches format and conventions without "
            "training; the fair baseline for a fine-tune; prompt grows with each shot's schema."
        ),
    )
    few_shot_seed: int = Field(
        13,
        description=(
            "What: seed for picking few-shot examples (combined with the question id). "
            "Effect: a different seed shows different examples."
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

    retry_on_error: bool = Field(
        False,
        description=(
            "What: when the first query fails to execute, send the SQLite error back once and "
            "score the second answer. Typical: false (plain) / true (retry row). Effect: fixes "
            "invented columns and syntax slips; one extra model call for failed questions only; "
            "the model never sees whether a query that ran was right."
        ),
    )


class EvalConfig(_Section):
    gold_timeout_s: float = Field(
        120.0,
        gt=0,
        description=(
            "What: max seconds for the gold (reference) query; the larger of this and "
            "exec_timeout_s is used. Typical: 120. Effect: slow but valid gold queries (some "
            "BIRD ones take ~26 s) are scored instead of randomly becoming gold_error."
        ),
    )
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
            "What: parallel worker processes executing SQL for scoring. Typical: CPU cores. "
            "Effect: speed only. Processes, not threads: SQLite's Python progress handler "
            "(the timeout) needs the GIL, so threads serialize and heavy queries time out."
        ),
    )
    mlflow_tracking_uri: str = Field(
        "sqlite:///mlflow.db",
        description=(
            "What: MLflow tracking store. Typical: sqlite:///mlflow.db (local). "
            "Effect: none on metrics; `mlflow ui --backend-store-uri sqlite:///mlflow.db`."
        ),
    )

    row_compare: Literal["multiset", "set"] = Field(
        "multiset",
        description=(
            "What: how result rows are compared. 'multiset' = duplicates count and ORDER BY in "
            "the gold query makes order matter (Spider convention); 'set' = set(gold) == "
            "set(pred), order and duplicates ignored (BIRD's official evaluator). Typical: "
            "multiset for Spider, set for BIRD."
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
    train_fraction: float = Field(
        1.0,
        gt=0.0,
        le=1.0,
        description=(
            "What: share of the training questions used (fixed shuffle with `seed`; smaller "
            "fractions are subsets of larger ones). Typical: 1.0; 0.125-0.5 for a learning "
            "curve. Effect: shows whether more data of the same kind would still help."
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
    optim: str = Field(
        "adamw_8bit",
        description=(
            "What: optimizer (transformers name). Typical: adamw_8bit (Unsloth default) or "
            "adamw_torch_fused. Effect: 8-bit optimizer states save ~0.5 GB VRAM on a 3B LoRA; "
            "quality is the same in practice."
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
    val_fraction: float = Field(
        0.0,
        ge=0.0,
        lt=0.5,
        description=(
            "What: share of each training source held out as validation, by whole database. "
            "Typical: 0 (no validation) or 0.05. Effect: eval_loss is logged every eval_steps "
            "to spot overfitting; those questions are not trained on."
        ),
    )
    eval_steps: int = Field(
        200,
        ge=1,
        description=(
            "What: compute validation loss every N optimizer steps (when val_fraction > 0). "
            "Typical: 200 (~5 points per epoch). Effect: each pass costs ~1-2 min."
        ),
    )
    train_sample_rows: int = Field(
        3,
        ge=0,
        le=10,
        description=(
            "What: example rows per table used for the training records that get rows (see "
            "train_sample_rows_mix). Typical: 3, same as the rows3 eval variant. Effect: must "
            "match the eval setting so the rows prompt is in-distribution."
        ),
    )
    train_sample_rows_mix: float = Field(
        0.0,
        ge=0.0,
        le=1.0,
        description=(
            "What: probability that a training record shows example rows (seeded per "
            "question). Typical: 0 (M0/M1a), 0.5 (M1b). Effect: the model sees both the plain "
            "and the rows prompt, so both eval variants match training; longer records."
        ),
    )
    longest_first: bool = Field(
        False,
        description=(
            "What: order training records longest first instead of shuffled. Typical: false; "
            "true only for a short VRAM probe (max_steps ~10). Effect: the first steps hit "
            "peak memory; never use for a real run (biased order)."
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
            "Effect: seed-to-seed EX variation is measured in M1 (seed2/seed3 runs)."
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
    test_once: bool = Field(
        False,
        description=(
            "What: allows loading a test set (Spider test, BIRD bird_holdout). "
            "Typical: false; true only in the generated final configs. "
            "Effect: without it the runner refuses test sets, so they are not looked at "
            "while models and prompts are still being chosen."
        ),
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
