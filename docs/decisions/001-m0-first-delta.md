# 001 · M0 first delta: QLoRA SFT vs prompt baseline (Spider dev)

Date: 2026-10-24. Runs: `results/m0_baseline/`, `results/m0_qlora/`, `results/m0_qlora_eval/`.

## Setup

- Model: Qwen2.5-Coder-3B-Instruct. The prompt is the same zero-shot one for both runs
  (`prompts/`), with greedy decoding (`temperature 0`, `--generation-config vllm`).
- Eval set: all 1034 Spider dev questions, scored by execution accuracy (EX) on SQLite. No
  gold query failed, so `n_gold_error = 0`.
- Fine-tune: QLoRA (NF4 base, r=16, alpha=16, lr 2e-4, 2 epochs, effective batch 16) on
  Spider train. 6918 records were used; 82 `baseball_1` records longer than 2048 tokens were
  dropped. Unsloth + TRL, with loss on the SQL completion only (checked on a real batch).
- Serving: the adapter was trained on the 4-bit base and is served by vLLM v0.31.0 on the
  **bf16** base (`--enable-lora`). M1 measures the merged / LoRA variants.

## Result

| | Baseline (prompt) | QLoRA SFT | Δ |
|---|---|---|---|
| EX, column order tolerant | 67.9 | **76.8** | **+8.9** |
| EX, strict column order | 64.2 | 73.6 | +9.4 |
| wrong_result | 25.1 | 21.5 | −3.7 |
| exec_error | 10.6 | 4.9 | −5.7 |
| latency p50 / p95 (ms, 16 concurrent) | 373 / 833 | 423 / 1029 | LoRA overhead |

Paired, strict EX: 151 fixed, 54 broken (column-tolerant EX: 144 fixed, 52 broken).
Exact McNemar gives p = 8.2e-12, but it assumes independent questions. Spider dev clusters
on 20 databases and contains paraphrase pairs, so the honest interval resamples databases:
**95% CI of the tolerant Δ = [+5.7, +11.7]** (DB-cluster bootstrap). The gain is not noise.

Training: 866 steps in 26.4 min (0.55 steps/s) on an RTX 5080. Peak VRAM was 3.5 GiB and
logged loss went from 0.36 to ~0.03; `final_train_loss` in `train_summary.json` (0.08) is the
average over the whole run. 1.7 % of the parameters are trainable.
Latency is indicative only: it comes from the eval run, not from a controlled benchmark
(the M1 bench will measure it properly).

## What changed

- **Fewer broken queries.** Of the 151 fixed questions, 56 had failed to execute before.
  Most of these baseline failures were alias mistakes in joins, e.g. selecting `T2.Country`
  where `T2` is the link table.
- **Spider's style was learned strongly.** 813 of the 1034 fine-tuned predictions copy
  Spider's `col  =  value` double-space formatting, against 21 for the baseline. 17 of the
  fixed questions differ from the baseline only in column order. That is why the
  column-tolerant delta (+8.9) is reported next to the strict one (+9.4): most of the gain
  is real, a small part is convention.
- **Broken questions show style over-applied.** Typical cases:
  - `count(*)` is put first when the question asks for the group first;
  - `DESC` is dropped ("ordered by age from the oldest");
  - Spider-like column names are invented (`pet_type` for `PetType`).

## Observations for M1

- The loss falls sharply at the epoch boundary (≈0.08 → 0.05), a sign of memorising
  examples on the second pass. Run an epoch sweep (1 / 2 / 3) on dev.
- Spider dev now guides decisions, so the headline must come from a set the fine-tune never
  saw in style: BIRD Mini-Dev. Report Spider test once at the end.
- The baseline's `exec_error` share (10.6 %) is large. Few-shot and retry-on-error rows may
  close part of the gap without training, so they are needed for a fair comparison.
- The plan's smoke evaluation on the 50-step adapter was skipped: vLLM was blocked by the
  missing NVIDIA Container Toolkit, and the full run was trained meanwhile. What it would
  have checked is covered: vLLM `/tokenize` gives prompt tokens identical to training on
  22 of 22 checks (`scripts/check_tokenize.py`), and the full eval ran with 0 transport
  errors (`n_transport_error` is now a metric).
- The tolerant EX column is the closer match to published Spider numbers: the official
  test-suite evaluator also accepts permuted columns. It is not that evaluator, though (no
  test-suite database variants), so "lucky" matches are not filtered out. The strict variant is kept because the
  gap between the two shows how much of the score is column-order convention.
