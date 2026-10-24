"""Export AWQ calibration samples for a merged fine-tune: chat messages, one JSON per line.

Purpose: AWQ picks per-channel scales from activations on sample inputs; in-domain samples
(our own SQL prompts + answers, as the model will see them) keep the 4-bit model closest to
the bf16 one. Validation records are used (unseen databases, both sources), so nothing here
is a test question.
Flow: training CONFIG -> build_sft_dataset -> seeded sample of the validation split (train if
there is none) -> outputs/calib/<run_name>.jsonl ({"messages": prompt + completion}).
Params: --config (the training config of the adapter), --n (samples, typical 256), --seed.
Try: `uv run python scripts/export_calib.py --config experiments/configs/m1b/train/mix7b.yaml`.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

from transformers import AutoTokenizer

from sqltune.config.loader import load_config
from sqltune.training.sft_data import build_sft_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--n", type=int, default=256)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    cfg = load_config(args.config)
    tokenizer = AutoTokenizer.from_pretrained(cfg.training.base_model)
    data = build_sft_dataset(cfg, tokenizer)
    pool = data.val if data.val is not None else data.train
    idx = random.Random(args.seed).sample(range(len(pool)), min(args.n, len(pool)))
    out = Path("outputs/calib") / f"{cfg.run.run_name}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        for i in idx:
            rec = pool[i]
            f.write(json.dumps({"messages": rec["prompt"] + rec["completion"]}) + "\n")
    print(f"wrote {len(idx)} samples to {out}")


if __name__ == "__main__":
    main()
