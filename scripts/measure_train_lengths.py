"""Token length of every training record (prompt + SQL) per source, with and without example
rows, to choose TrainConfig.max_seq_length before a long run.

Purpose: Records longer than max_seq_length are dropped, not truncated. BIRD schemas are wide,
and the dropped records come from the widest (hardest) databases, so the cap is chosen to keep
~all of BIRD (plan M1b, revision 1) and the cost is paid in VRAM, not in biased data.
Flow: load_spider(train) / load_bird_train() -> to_sft_record(rows 0 and rows 3) -> Qwen
tokenizer -> percentiles and drops per cap (and the databases that would lose records).
Params: CAPS, ROWS below.
Try: `uv run --extra train python scripts/measure_train_lengths.py`.
"""

from __future__ import annotations

from collections import Counter

from transformers import AutoTokenizer

from sqltune.config.models import DataConfig, PromptConfig
from sqltune.data.bird_train import load_bird_train
from sqltune.data.spider import load_spider
from sqltune.training.sft_data import _n_tokens, to_sft_record

CAPS = [2048, 3072, 4096, 5120, 6144, 8192]
ROWS = [0, 3]


def main() -> None:
    tok = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-Coder-3B-Instruct")
    cfg = DataConfig()
    sources = {"spider": load_spider(cfg, "train"), "bird": load_bird_train(cfg)}
    for source, examples in sources.items():
        for rows in ROWS:
            prompt = PromptConfig(sample_rows=rows)
            lengths = []
            for ex in examples:
                record = to_sft_record(ex, prompt)
                lengths.append((_n_tokens(tok, record["prompt"] + record["completion"]), ex.db_id))
            values = sorted(n for n, _ in lengths)
            n = len(values)
            print(
                f"\n{source} rows={rows}: n={n} p50={values[n // 2]} "
                f"p95={values[int(0.95 * (n - 1))]} p99={values[int(0.99 * (n - 1))]} max={values[-1]}"
            )
            for cap in CAPS:
                over = Counter(db for length, db in lengths if length > cap)
                total = sum(over.values())
                print(
                    f"  cap {cap}: drop {total} ({100 * total / n:.2f}%) from {len(over)} DBs "
                    f"{dict(over.most_common(5)) if over else ''}"
                )


if __name__ == "__main__":
    main()
