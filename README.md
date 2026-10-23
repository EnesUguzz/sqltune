# sqltune

Fine-tuning Qwen2.5-Coder-3B-Instruct for text-to-SQL (SQLite) with QLoRA, served with vLLM and
measured by execution accuracy on Spider and BIRD Mini-Dev.

> Work in progress (M0: first baseline → QLoRA delta on Spider dev).

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

_Filled in during M0 Task 2._
