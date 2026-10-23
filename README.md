# sqltune

Fine-tuning Qwen2.5-Coder-3B-Instruct for text-to-SQL (SQLite) with QLoRA, served with vLLM and
measured by execution accuracy on Spider and BIRD Mini-Dev.

> Work in progress. **M0 result (Spider dev, 1034 questions):** QLoRA SFT lifts
> Qwen2.5-Coder-3B from **67.9 → 76.8 EX** (+8.9, 95% CI [+5.7, +11.7] by database; column-order tolerant like the official
> evaluator; strict column order 64.2 → 73.6). 151 questions fixed / 54 broken, McNemar
> p ≈ 1e-11, after 26 min of training at 3.5 GiB peak VRAM. Details:
> [docs/decisions/001-m0-first-delta.md](docs/decisions/001-m0-first-delta.md).

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
