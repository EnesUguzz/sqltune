# sqltune

Fine-tuning Qwen2.5-Coder-3B-Instruct for text-to-SQL (SQLite) with QLoRA, served with vLLM and
measured by execution accuracy on Spider and BIRD Mini-Dev.

**Result.** A 3B model fine-tuned with QLoRA on Spider + BIRD train (15k examples, 93 min on
one 16 GB GPU, 3.9 GiB peak VRAM) **matches the 7B model of the same family (AWQ 4-bit) on BIRD** and gains
+10 EX over its own base. Numbers are on held-out test sets, each run exactly once:

| model | variant | BIRD holdout EX | Spider test EX | train time / VRAM |
|---|---|---|---|---|
| Qwen2.5-Coder-3B base | zero-shot | 40.6 | 71.7 | — |
| Qwen2.5-Coder-3B base | rows + retry | 46.2 | 76.6 | — |
| 3B + Spider-only QLoRA (M0) | zero-shot | 36.1 | 77.7 | 26 min / 3.5 GiB |
| 3B + Spider-only QLoRA (M0) | rows + retry | 39.8 | 80.2 | |
| **3B + Spider+BIRD QLoRA (m1b_mix)** | zero-shot | **50.6** | 78.6 | 93 min / 3.9 GiB |
| **3B + Spider+BIRD QLoRA (m1b_mix)** | rows + retry | **52.2** | 80.9 | |
| Qwen2.5-Coder-7B AWQ | zero-shot | 46.6 | 78.7 | — |
| Qwen2.5-Coder-7B AWQ | rows + retry | 51.4 | 82.6 | — |
| Qwen2.5-Coder-14B AWQ | zero-shot | 54.2 | 82.5 | — |
| Qwen2.5-Coder-14B AWQ | rows + retry | 60.2 | 85.8 | — |

EX = execution accuracy, column-order tolerant (strict numbers and every other run:
[results/SUMMARY.md](results/SUMMARY.md)). *rows + retry*: the prompt shows 3 example rows per
table, and a query that fails to execute is retried once with the error message.
BIRD holdout = 250 Mini-Dev questions on 6 databases never used for any decision.

- **BIRD train is what helps on BIRD.** m1b_mix vs base: +10.0 (zero-shot, McNemar p = 3e-4)
  and +6.0 (rows + retry, p = 0.02). The Spider-only adapter is *below* the base on BIRD.
- **3B fine-tune ≈ 7B AWQ base on BIRD** (52.2 vs 51.4, p = 0.89), 1.7 points behind it on Spider.
  14B is still clearly ahead.
- **Spider is not hurt** by adding BIRD (80.9 vs 80.2 for the Spider-only adapter, p = 0.25).

How it got there, one decision note per milestone:
[001](docs/decisions/001-m0-first-delta.md) first fine-tune on Spider ·
[002](docs/decisions/002-m1a-bird-baselines-sweeps.md) BIRD baselines and training sweeps ·
[003](docs/decisions/003-m1b-bird-train.md) adding BIRD train, error analysis, test runs.

### Caveats

- BIRD holdout is a 6-database subset of Mini-Dev; numbers are not directly comparable with
  published Mini-Dev or BIRD-dev leaderboards.
- Spider test EX comes from this repo's harness, not the official test-suite evaluation.
- The 3B rows are served in bf16 (base + LoRA adapter); 7B and 14B are the official AWQ
  4-bit checkpoints (they do not fit the 16 GB card in bf16).
- The M0 adapter trained on Spider `soccer_1`, which holds the same data as the holdout
  database `european_football_2`; it still scores below the base on BIRD holdout.
- Qwen2.5-Coder may have seen Spider / BIRD in pre-training. Absolute scores may be inflated;
  differences against the same base model are still valid.
- BIRD gold SQL has known semantic errors. In a hand-labelled sample of 40 m1b_mix failures,
  6 were gold problems or ambiguous questions (003).

## Experiment tracking

Every training and eval run is logged to MLflow (`sqlite:///mlflow.db`, local; the database
is not in the repo, the run folders under `results/` are). Run `uv run mlflow ui` to browse.

- **Training runs:** the full config, train loss per 10 steps, validation loss on held-out
  databases every 200 steps, runtime, peak VRAM, and per-source counts (records, validation
  records, records dropped as too long, rows-prompt fallbacks). `results/<run>/train_summary.json`
  keeps the same numbers plus the MLflow run id.
- **Eval runs:** the full config, EX (strict and tolerant) with a database-level bootstrap CI,
  outcome counts by error type, latency and token counts, and the predictions file. Each eval
  run is tagged with the training run it evaluates (`train_run`, `train_mlflow_run_id`), the
  prompt variant and the eval set, so any number can be traced back to its adapter.

![Training loss](docs/img/m1b_loss.png)

Validation loss on unseen databases flattens after ~1/5 of the epoch (0.209 → 0.201): the
format is learned early. M0's train loss falls lower because it ran 2 epochs on Spider only.

![Learning curve](docs/img/m1_learning_curve.png)

More Spider data keeps helping Spider but never lifts BIRD above the base model; BIRD-style
data did (m1b_mix: 46.0 on bird_diag vs 33.6 for base).

![MLflow run comparison](docs/img/mlflow_compare.png)

MLflow's compare view for M0, the 1-epoch Spider run and m1b_mix: parameters that differ
and final metrics.

Prompt-token parity between training and vLLM: `uv run --extra train python
scripts/check_tokenize.py` (with `make serve LORA=m0_qlora` running) → 22/22 identical.

## Environment

Verified on an RTX 5080 16 GB (Blackwell, sm_120), driver 610.88, WSL2 Ubuntu 24.04, Python 3.12:

| Package | Version |
|---|---|
| torch | 2.14.1+cu130 |
| unsloth / unsloth-zoo | 2026.10.8 |
| bitsandbytes | 0.50.2 |
| transformers | 5.17.0 |
| trl | 1.15.0 |
| peft | 0.21.2 |
| vLLM (Docker) | v0.31.0 |

Setup: `uv sync --extra train`, then `make check-env`. Unsloth warns that the bundled xformers
wheel does not match torch 2.14 and falls back to PyTorch SDPA attention; that is harmless.

Docker needs the NVIDIA Container Toolkit for GPU access (`docker_gpu` in `make check-env`).

## Data

`make download` fetches the official Spider zip (2024 release, CC BY-SA 4.0) from Google Drive
and unpacks it into `data/spider/` (1.8 GB, git-ignored).

| Split | Questions | Used for |
|---|---|---|
| train (`train_spider.json`) | 7000 | SFT |
| train_others | 1659 | not used in M0 |
| dev | 1034 | baseline vs fine-tune comparison |
| test | 2147 | final numbers only (run once) |

166 SQLite databases under `database/` cover train and dev; dev databases do not appear in
train. If gdown hits Drive's quota, download the zip in a browser, save it as
`data/raw/spider.zip` and re-run `make download`.

**BIRD Mini-Dev (SQLite).** `make download` also fetches it: the questions come from
Hugging Face (`birdsql/bird_mini_dev`, the canonical copy) and the 11 SQLite databases from
the official zip (1.4 GB in `data/bird/`). It has 500 questions: 148 simple, 250 moderate and
102 challenging. Each question comes with an *evidence* hint, and that hint goes into the
prompt (BIRD's standard setup). The set is split by database:

| Half | Questions | Databases | Used for |
|---|---|---|---|
| `diag` | 250 | california_schools, card_games, formula_1, superhero, thrombosis_prediction | decisions, diagnostics |
| `holdout` | 250 | codebase_community, debit_card_specializing, european_football_2, financial, student_club, toxicology | headline numbers, once per final model |

The split is pinned in `DataConfig.bird_diag_dbs` (chosen once as the database subset closest
to 250 questions), so no seed or other setting can move holdout questions into diag.

**BIRD train (SQLite).** `make download` also fetches BIRD train (8.9 GB zip, 69 databases,
9,428 questions, `data/bird_train/`). M1b trains on it after these checks
(`make dataset-review`, notes in [docs/data/dataset-findings.md](docs/data/dataset-findings.md)):

- Databases with the same data as an eval database are removed: BIRD `world` and `airline`
  (Spider dev), and Spider train `formula_1` / `soccer_1` (BIRD Mini-Dev). Overlap is found by
  comparing table names, since the same public data appears under different database names.
- 402 questions whose gold SQL fails or returns > 100k rows are skipped (330 of them from
  `retail_world`).
- 5% of each source is held out as whole databases for validation loss.

Gold queries get their own timeout (`gold_timeout_s`, 120 s), because some BIRD gold
queries take ~26 s and would otherwise flip to `gold_error` from run to run (this happened
to `bird-0393` in 7 of the 20 M1a `bird_diag` runs, before this setting existed). BIRD runs
use `row_compare: set` and `exec_timeout_s: 30`. BIRD's official evaluator also compares sets
of rows but keeps column order, so the **strict** EX (in brackets in the tables) is the number
comparable to published BIRD results; the headline EX additionally tolerates column order.
