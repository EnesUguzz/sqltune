"""Environment sanity check: torch + CUDA, bf16, bitsandbytes 4-bit, Unsloth, Docker GPU.

Purpose: The RTX 5080 (Blackwell, sm_120) needs CUDA 12.8+ builds of torch and
bitsandbytes >= 0.45; this catches a wrong build before an hour-long training run fails.
Flow: `sqltune check-env` -> gpu_report() + docker_gpu_ok() -> printed table.
Params: none.
Try: Run it after any torch / unsloth / bitsandbytes upgrade.
"""

from __future__ import annotations

import subprocess
from typing import Any

MATMUL_SIZE = 1024
DOCKER_CUDA_IMAGE = "nvidia/cuda:12.8.0-base-ubuntu24.04"


def gpu_report() -> dict[str, Any]:
    """Probe the training stack; every failure is reported as a value, never raised."""
    info: dict[str, Any] = {}
    try:
        import torch
    except ImportError as exc:
        return {"torch": f"missing ({exc}); run `uv sync --extra train`"}

    info["torch"] = torch.__version__
    info["torch_cuda"] = torch.version.cuda
    info["cuda_available"] = torch.cuda.is_available()
    if not info["cuda_available"]:
        return info
    info["device"] = torch.cuda.get_device_name(0)
    info["capability"] = torch.cuda.get_device_capability(0)
    info["bf16_supported"] = torch.cuda.is_bf16_supported()
    a = torch.randn(MATMUL_SIZE, MATMUL_SIZE, device="cuda", dtype=torch.bfloat16)
    info["bf16_matmul_ok"] = bool(torch.isfinite(a @ a).all().item())

    try:
        import bitsandbytes as bnb

        info["bitsandbytes"] = bnb.__version__
        layer = bnb.nn.Linear4bit(  # type: ignore[attr-defined,no-untyped-call]
            256, 256, compute_dtype=torch.bfloat16, quant_type="nf4"
        )
        layer = layer.to("cuda")
        out = layer(torch.randn(4, 256, device="cuda", dtype=torch.bfloat16))
        info["nf4_matmul_ok"] = bool(torch.isfinite(out).all().item())
    except Exception as exc:
        info["nf4_matmul_ok"] = f"failed: {exc}"

    try:
        import unsloth

        info["unsloth"] = unsloth.__version__
    except Exception as exc:
        info["unsloth"] = f"failed: {exc}"

    for name in ("transformers", "trl", "peft"):
        try:
            info[name] = __import__(name).__version__
        except Exception as exc:
            info[name] = f"failed: {exc}"
    return info


def docker_gpu_ok() -> str:
    """Run nvidia-smi inside a CUDA container; returns 'ok' or the error text."""
    cmd = ["docker", "run", "--rm", "--gpus", "all", DOCKER_CUDA_IMAGE, "nvidia-smi", "-L"]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=300, check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        return f"failed: {exc}"
    if proc.returncode != 0:
        return f"failed: {proc.stderr.strip()[-300:]}"
    return "ok: " + proc.stdout.strip()
