# 004 — M1b: the same fine-tune on the 7B base

Date: 2026-10-25. Branch `m1b`. Follows [003](003-m1b-bird-train.md). Dev sets only;
the test sets were not touched.

## Question

On 3B, Spider + BIRD train gave +10 to +12 EX on BIRD over the base (003). Does the same data
on top of Qwen2.5-Coder-7B-Instruct close the gap to 14B?

## What was run

- **Training:** `m1b/train/mix7b.yaml`. Same data, split, LoRA settings and 1 epoch as
  `m1b_mix`; only the base model changes. 932 steps, 3 h 4 min, peak 8.6 GiB VRAM, final
  validation loss 0.113 (3B: 0.200).
- **Serving.** The 7B does not fit the 16 GB card in bf16 under vLLM, so the LoRA is merged
  into the bf16 base and the result is quantized. vLLM v0.31 no longer serves bitsandbytes,
  so the NF4 training base cannot be used directly. Two paths:
  - **AWQ 4-bit** with `scripts/merge_awq.py` (llm-compressor, W4A16 asymmetric, group 128,
    128 calibration samples from the validation split). The base 7B is quantized with the
    same script and data (`qwen7b_own`), so the pair differs only by the fine-tune.
  - **FP8 check** (`outputs/fp8check.sh`): the merged bf16 checkpoint and the bf16 base,
    both served with vLLM online FP8 (8-bit weights, near lossless). This tests whether
    4-bit quantization erased the gain.
- **AWQ calibration size matters for speed, not just quality.** A first run used 256 samples
  of up to 2,048 tokens (~190k tokens). One layer's cached activations then no longer fit in
  16 GB and spilled into shared memory, which was about 300 times slower (estimated 8+ hours).
  128 samples of up to 1,024 tokens (~67k tokens) took 6 minutes per model.

## Dev results (EX %, column-order tolerant; strict in brackets)

| model | variant | bird_diag | Spider dev |
|---|---|---|---|
| 7B base, official AWQ | zero | 45.6 (44.8) | 78.7 (74.7) |
| 7B base, official AWQ | rows3 + retry | 51.6 (50.8) | 82.4 (78.1) |
| 7B base, own AWQ | zero | 46.4 (45.2) | 77.9 (74.1) |
| 7B base, own AWQ | rows3 + retry | 51.6 (50.0) | 81.6 (77.9) |
| 7B fine-tune, own AWQ | zero | 46.8 (45.6) | 80.7 (77.9) |
| 7B fine-tune, own AWQ | rows3 + retry | 52.0 (51.2) | 83.1 (80.7) |
| 7B base, FP8 | zero | 48.8 (48.4) | 78.1 (74.3) |
| 7B base, FP8 | rows3 + retry | 53.2 (52.8) | 83.0 (78.9) |
| 7B fine-tune, FP8 | zero | 48.8 (48.0) | 82.0 (79.3) |
| 7B fine-tune, FP8 | rows3 + retry | 53.2 (52.4) | 83.3 (80.6) |
| 3B fine-tune (`m1b_mix`, bf16) | rows3 + retry | 48.8 (48.4) | 78.6 (75.7) |
| 14B AWQ | retry | 57.6 (56.8) | 82.6 (77.9) |

Paired exact McNemar:

| comparison | set | Δ | fixed / broken | p |
|---|---|---|---|---|
| 7B fine-tune vs base, AWQ, zero | bird_diag | +0.4 | 25 / 24 | 1.0 |
| 7B fine-tune vs base, AWQ, rows3 + retry | bird_diag | +0.4 | 26 / 25 | 1.0 |
| 7B fine-tune vs base, FP8, zero | bird_diag | 0.0 | 24 / 24 | 1.0 |
| 7B fine-tune vs base, FP8, rows3 + retry | bird_diag | 0.0 | 22 / 22 | 1.0 |
| 7B fine-tune vs base, AWQ, zero | Spider dev | +2.8 | 77 / 48 | 0.012 |
| 7B fine-tune vs base, FP8, zero | Spider dev | +3.9 | 85 / 45 | 6e-4 |
| 7B fine-tune vs base, FP8, rows3 + retry | Spider dev | +0.3 | 46 / 43 | 0.83 |
| own AWQ vs official AWQ (base), rows3 + retry | bird_diag | 0.0 | 7 / 7 | 1.0 |
| FP8 vs own AWQ (base), zero | bird_diag | +2.4 | 10 / 4 | 0.18 |
| 3B fine-tune vs 7B official AWQ, rows3 + retry | bird_diag | −2.8 | 27 / 34 | 0.44 |
| 7B fine-tune (AWQ) vs 14B, retry vs rows3 + retry | bird_diag | −5.6 | 23 / 37 | 0.09 |

## Findings

1. **The fine-tune gives the 7B nothing on BIRD.** Δ is +0.4 with AWQ and exactly 0.0 with
   FP8. That is far under the pre-registered 5-point rule and not significant. The model does
   change about a fifth of its BIRD answers (~25 fixed, ~25 broken). Execution errors drop
   (32 → 20 of 250, zero-shot AWQ), but the extra queries that now run return wrong results.
   So the 7B learned the training format and schema style, but not more correct BIRD SQL.
2. **It is not a quantization artefact.** The FP8 pair shows the same null result. Our own
   AWQ matches the official Qwen AWQ checkpoint (0.0 on BIRD rows3 + retry, −0.8 on Spider,
   neither significant). AWQ 4-bit costs about 2 points against FP8 on BIRD zero-shot, but
   this is not significant (p = 0.18).
3. **The only gain is Spider zero-shot** (+3.9, p = 6e-4). It disappears once the prompt shows
   example rows and retries errors (+0.3, p = 0.83). This looks like format learning (Spider
   gold style, value spelling) that a better prompt also gives.
4. **Why the 3B gained and the 7B did not.** This is our reading, not tested. The 3B base
   makes many schema and format errors that 15k examples fix: base `no such column` share
   30% → 14% (003). The 7B base is already past that point. Its remaining BIRD errors
   (evidence use, logic, gold noise; see the error classes in 003) are not what one epoch
   of r = 16 LoRA on this data teaches. The lower validation loss (0.113 vs 0.200) fits this
   reading: the 7B fits the training distribution better, but that does not carry over to
   Mini-Dev databases.
5. **Fair dev comparison for 003 finding 2.** With the same variant (rows3 + retry), the 3B
   fine-tune is 2.8 points below the 7B on `bird_diag` (p = 0.44). On the test set it was
   0.8 above (p = 0.89). Both say "level with the 7B within noise", not "better".

## Decision

- No 7B test-set run is needed to support a claim. The dev result is a clear null on BIRD.
  It stays a dev finding unless the test set is run (only with the user's approval).
- Headline model stays the 3B `m1b_mix`. The project result is "a 3B QLoRA reaches the
  7B base", and "the same recipe does not lift the 7B" is the honest second half.
- Options to lift the 7B, none started: more epochs or a higher LoRA rank; harder or
  cleaner BIRD data; schema pruning / linking at inference time (the largest error class in
  003); or RL on execution feedback (GRPO, planned as optional M2).

## Caveats

- `bird_diag` is 250 questions on 5 databases; a 2-point difference is about 5 questions.
- The adapter was trained on the NF4 base and merged into bf16 weights. The merged model may
  behave slightly differently from the trained one. We could not measure the NF4 + LoRA
  model directly (no bitsandbytes in this vLLM). The FP8 and AWQ paths agree, so this is
  probably small.
- FP8 runs are not part of the test-once matrix; they exist only for this check.
