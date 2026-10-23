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

 then uv run sqltune check-env
🦥 Unsloth: Will patch your computer to enable 2x faster free finetuning.
Unsloth: xformers is installed but cannot load with torch 2.14.1+cu130 (it was built for 2.10.0+cu128), so PyTorch attention is used instead.
To install a matching xformers, run `unsloth install-kernels xformers`, or `pip uninstall xformers` to hide this. UNSLOTH_ENABLE_LOGGING=1 shows the full error.
🦥 Unsloth Zoo will now patch everything to make training faster!
torch           2.14.1+cu130
torch_cuda      13.0
cuda_available  True
device          NVIDIA GeForce RTX 5080
capability      (12, 0)
bf16_supported  True
bf16_matmul_ok  True
bitsandbytes    0.50.2
nf4_matmul_ok   True
unsloth         2026.10.8
transformers    5.17.0
trl             1.15.0
peft            0.21.2
docker_gpu      failed: mplete
Digest: sha256:e778509d37d66475120929671500377524f7278478ba08131b07ef3ffcc0dce0
Status: Downloaded newer image for nvidia/cuda:12.8.0-base-ubuntu24.04
docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]]

Run 'docker run --help' for more information. Unsloth warns that the bundled xformers wheel does
not match torch 2.14 and falls back to PyTorch SDPA attention; that is harmless.

## Data

_Filled in during M0 Task 2._
