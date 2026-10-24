# sqltune

Fine-tuning Qwen2.5-Coder-3B-Instruct for text-to-SQL (SQLite) with QLoRA, served with vLLM and
measured by execution accuracy on Spider and BIRD Mini-Dev.

> Work in progress. **M0 result (Spider dev, 1034 questions):** QLoRA SFT lifts
> Qwen2.5-Coder-3B from **67.9 → 76.8 EX** (+8.9, 95% CI [+5.7, +11.7] by database; column-order tolerant like the official
> evaluator; strict column order 64.2 → 73.6). 151 questions fixed / 54 broken, McNemar
> p ≈ 1e-11, after 26 min of training at 3.5 GiB peak VRAM. Details:
> [docs/decisions/001-m0-first-delta.md](docs/decisions/001-m0-first-delta.md).
>
> **M1a (BIRD Mini-Dev diag split, 250 questions):** the Spider-only fine-tune does not
> help on BIRD — every adapter is at or below the base model (33.7 EX), though part of this may
> be that adapters never saw BIRD's evidence line in training; example rows + retry give +4.8
> with no training; 7B / 14B AWQ (with retry) reach 48.4 / 57.6. On Spider, the full-data
> adapters (rank, epochs, seeds) all land within 76.2–77.9, about the seed-to-seed spread. Details:
> [docs/decisions/002-m1a-bird-baselines-sweeps.md](docs/decisions/002-m1a-bird-baselines-sweeps.md),
> table: [results/SUMMARY.md](results/SUMMARY.md).

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

Gold queries get their own timeout (`gold_timeout_s`, 120 s), because some BIRD gold
queries take ~26 s and would otherwise flip to `gold_error` from run to run (this happened
to `bird-0393` in 7 of the 20 M1a `bird_diag` runs, before this setting existed). BIRD runs
use `row_compare: set` and `exec_timeout_s: 30`. BIRD's official evaluator also compares sets
of rows but keeps column order, so the **strict** EX (in brackets in the tables) is the number
comparable to published BIRD results; the headline EX additionally tolerates column order.
