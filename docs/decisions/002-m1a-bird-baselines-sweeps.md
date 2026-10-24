# 002 — M1a: BIRD baselines, prompt variants and training sweeps

Date: 2026-10-24. Branch `m1a`. Full table: `results/SUMMARY.md` (`make summarize`).

## What was measured

40 eval runs on two decision sets: Spider dev (1034 questions) and `bird_diag` (250 BIRD
Mini-Dev questions from 5 databases; the other 6 databases form `bird_holdout` and were **not**
touched). Spider test and `bird_holdout` stay unused until the final M1b model.

- Base 3B (Qwen2.5-Coder-3B-Instruct) × 5 prompt variants: zero, 3 few-shot examples,
  3 example rows per table, one retry on execution error, rows + retry.
- 10 adapters, all trained on Spider train only: M0, learning curve (1/8, 1/4, 1/2, all),
  `others` (+ train_others), `ep3` (3 epochs), `lora16` (rank 16), seeds 2 and 3.
- Qwen2.5-Coder 7B and 14B (AWQ 4-bit), zero and retry.

EX = execution accuracy, column-order tolerant (strict, comparable to published BIRD numbers, is in `results/SUMMARY.md`). Δ = against base/zero on the same set with a
95% CI from resampling databases.

## Results (EX %)

| model / variant | Spider dev | bird_diag |
|---|---|---|
| base 3B, zero | 68.1 | 33.7 |
| base 3B, rows3 + retry | 73.2 (+5.1) | **38.6 (+4.8)** |
| best 3B adapter (seed2 / lora16 / lc_f1) | ~77.5 (+9.4) | 32–33 (−1) |
| 3B adapter range (10 adapters, zero-shot) | 73.1–77.9 | 28.4–32.9 |
| 7B AWQ, retry | 80.1 (+12.0) | 48.4 (+14.7) |
| 14B AWQ, retry | **82.6 (+14.5)** | **57.6 (+23.9)** |

## Findings

1. **The Spider-only fine-tune does not help on BIRD.** All 10 adapters (zero-shot) land
   at or below the base model on `bird_diag` (−0.8 to −5.3; only `others` has a CI excluding
   zero). The M0 gain is a Spider-style gain, not a general text-to-SQL gain. Caveat: the
   adapters never saw BIRD's `Evidence:` line during training, so part of the drop may be a
   prompt-format shift rather than missing skill; M1b (BIRD train with evidence) tests this.
2. **More Spider data has diminishing returns.** Learning curve on Spider: 1/8 → 73.1,
   1/4 → 75.5, 1/2 → 75.5, all → 77.3 (+4.2 from 1/8 to all, +1.8 from 1/2 to all). Adding
   `train_others` (75.6) and 3 epochs (77.2) did not help.
3. **The sweep differences are inside seed noise.** Three seeds of the same config give
   76.8 / 77.9 / 76.2 on Spider (spread 1.7). No hyperparameter choice (rank, epochs, extra
   data) is distinguishable from that.
4. **Invented columns remain the main error, and fine-tuning only halves them on Spider.**
   `no such column` share: Spider base 9.6% → adapters 3.6–7.0%; BIRD base 31.6% → adapters
   22.4–31.2% (zero-shot; 21.6% for M0 with retry). Example rows + retry bring BIRD to 24.8%; 14B to 0.4%.
5. **Cheap inference-time tricks help on both sets.** rows3 + retry: +5.1 Spider, +4.8 BIRD,
   no training. Few-shot from Spider train does not help (+1.1 / −1.3) and makes prompts much longer.
6. **Model size beats everything we did to 3B on BIRD.** 7B zero-shot already beats every
   3B variant on both sets; 14B is +24 on BIRD.

## Run notes

- A power cut interrupted the training sweep at `lora16`; the half-finished run was deleted
  and `lora16`, `seed2`, `seed3` were retrained from scratch.
- A game was running on the same GPU for the 3B training of `lora16`/`seed2`/`seed3`
  (28–40 min instead of ~13) and for most 3B evals: **latencies (p50 ms) in the table are not
  comparable across runs**; accuracy is unaffected (greedy decoding).
- 14B did not start with the default server settings (9.9 GiB weights + 1.8 GiB CUDA graphs
  left 0.65 GiB KV cache). It ran with `GPU_UTIL=0.88` and `--enforce-eager` (no CUDA graphs,
  3.58 GiB KV cache), which makes its latency higher than it would otherwise be.
- `m0_qlora` zero on Spider reproduced M0 exactly (76.8), so the M1 harness matches M0.
  Base zero gave 68.1 vs 67.9 in M0 (2 questions; cause not investigated).
- No transport errors in any run.
- `bird_diag` has only 5 databases, so its database-resampled CIs are indicative only
  (too few clusters for a reliable percentile bootstrap); Spider dev has 20.
- `bird-0393` (gold ~26 s vs the 30 s timeout) was `gold_error` in 7 of the 20 `bird_diag`
  runs, so those runs score 249 questions instead of 250 (≤0.4 points). Fixed after M1a with
  a separate `gold_timeout_s` (120 s); M1a numbers were not re-run.
- The diag/holdout split used to depend on `data.seed` (seed 14 would have swapped the halves).
  It is now pinned in `DataConfig.bird_diag_dbs`; M1a used the same databases as listed here.
- nsc divides by all questions, exec_error by scorable ones, so they differ slightly on
  `bird_diag`.

## Suggested direction (decided on Spider dev + bird_diag only)

The question for M1b is no longer "which hyperparameters" — they are within noise. It is
"what training data makes a 3B model generalise beyond Spider". Options, cheapest first:

1. Keep rows3 + retry as the default inference setting for every model (free +5).
2. Train on data that looks like BIRD: BIRD train (9.4k questions, with evidence) plus Spider,
   then re-measure on `bird_diag`. This tests whether the BIRD drop is about data distribution.
3. Only if (2) is not enough: synthetic / column-renamed data (the Haiku option, after the
   quota check).
4. Report 7B/14B as the reference ceiling; a 3B fine-tune that closes part of the gap to 7B
   on BIRD is the portfolio story.

Final config for test-once (Spider test, `bird_holdout`) is not chosen yet.

## Questions for the morning

1. Haiku quota — after measuring, decide whether synthetic data is in scope for M1b.
2. M1b scope: add BIRD train to the training data (option 2)?
3. Merge `m0` and `m1a` into `main`, and push?
