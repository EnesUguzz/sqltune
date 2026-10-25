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
  - **FP8 check** (`scripts/oneoff/fp8check.sh`; `merge_awq.py --no-quantize`): the merged
    bf16 checkpoint and the bf16 base,
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

Paired exact McNemar (`scripts/paired_stats.py dev7b`; 95% CI on Δ at question level):

| comparison | set | Δ | fixed / broken | p | 95% CI |
|---|---|---|---|---|---|
| 7B fine-tune vs base, AWQ, zero | bird_diag | +0.4 | 25 / 24 | 1.0 | −5.1 to +5.9 |
| 7B fine-tune vs base, AWQ, rows3 + retry | bird_diag | +0.4 | 26 / 25 | 1.0 | −5.2 to +6.0 |
| 7B fine-tune vs base, FP8, zero | bird_diag | 0.0 | 24 / 24 | 1.0 | −5.4 to +5.4 |
| 7B fine-tune vs base, FP8, rows3 + retry | bird_diag | 0.0 | 22 / 22 | 1.0 | −5.2 to +5.2 |
| 7B fine-tune vs base, AWQ, zero | Spider dev | +2.8 | 77 / 48 | 0.012 | +0.7 to +4.9 |
| 7B fine-tune vs base, FP8, zero | Spider dev | +3.9 | 85 / 45 | 6e-4 | +1.7 to +6.0 |
| 7B fine-tune vs base, FP8, rows3 + retry | Spider dev | +0.3 | 46 / 43 | 0.83 | −1.5 to +2.1 |
| own AWQ vs official AWQ (base), rows3 + retry | bird_diag | 0.0 | 7 / 7 | 1.0 | −2.9 to +2.9 |
| FP8 vs own AWQ (base), zero | bird_diag | +2.4 | 10 / 4 | 0.18 | −0.5 to +5.3 |
| 3B fine-tune vs 7B official AWQ, rows3 + retry | bird_diag | −2.8 | 27 / 34 | 0.44 | −8.9 to +3.3 |
| 3B fine-tune vs 7B FP8, rows3 + retry | bird_diag | −4.4 | 22 / 33 | 0.18 | −10.2 to +1.4 |
| 3B fine-tune vs 7B FP8, rows3 + retry | Spider dev | −4.4 | 58 / 103 | 5e-4 | −6.7 to −2.0 |
| 7B fine-tune (AWQ) vs 14B, retry vs rows3 + retry | bird_diag | −5.6 | 23 / 37 | 0.09 | |

**3B gain minus 7B gain** on the same `bird_diag` questions (`paired_stats.py did`):
zero-shot +12.4 vs 0.0 (difference +12.4, 95% CI +5.2 to +19.6); rows3 + retry +10.4 vs
0.0 (difference +10.4, CI +3.3 to +17.5). The two gains differ significantly, even though
the 7B gain alone cannot be pinned below ~5 points.

## Did the 7B learn anything? (added 2026-10-25, after an independent review)

The dev null could mean "the 7B learned nothing" or "it learned, but the gain does not carry
over to Mini-Dev". Three checks, none on test sets:

**Validation loss** (`scripts/val_loss.py`, completion tokens of the 838 held-out validation
records, 9 databases never trained on; token-weighted):

| | base | fine-tuned | drop |
|---|---|---|---|
| 3B | 0.333 | 0.204 | −39% |
| 7B | 0.218 | 0.119 | −45% |

The 7B learned the training distribution at least as much as the 3B, in loss terms. The 7B
base is already where the fine-tuned 3B ends up.

**Execution accuracy on BIRD train-distribution questions** (`scripts/eval_in_distribution.py`,
zero-shot; 7B in FP8; `paired_stats.py indist`):

| set | 3B base | 3B fine-tune | Δ (p) | 7B base | 7B fine-tune | Δ (p) |
|---|---|---|---|---|---|---|
| 250 training questions (seen in training) | 48.4 | 64.8 | +16.4 (5e-8) | 64.4 | 71.6 | +7.2 (0.01) |
| 472 validation questions (3 unseen BIRD-train DBs) | 36.0 | 46.8 | +10.8 (7e-7) | 56.4 | 58.5 | +2.1 (0.25) |
| 250 `bird_diag` questions (Mini-Dev) | 33.6 | 46.0 | +12.4 (9e-5) | 48.8 | 48.8 | 0.0 (1.0) |

For the 3B the gain is about the same everywhere: on questions it trained on, on unseen
databases of the same distribution and on Mini-Dev. For the 7B it shrinks step by step:
+7 on its own training questions, +2 on unseen databases of the same distribution, 0 on
Mini-Dev. Even on questions it was trained on, the 7B gains only 7 points, while its loss
drops 45%. So most of what the 7B learned is how the gold SQL is written (aliases, layout,
which of several equivalent forms), which lowers loss but rarely changes the result.

**Can a prompt replace the 3B fine-tune?** The independent review noted that half of the
3B's BIRD gain comes from execution errors that disappear (SQLite dialect, unquoted names
with spaces, columns from the wrong table). We wrote a system prompt with SQLite rules
(`prompts/sql_system_sqlite_rules.md`: quote identifiers with backticks, prefix columns with
their table, `strftime` instead of `YEAR()`, `CAST` before dividing, ...) and ran the base 3B
with it (`experiments/configs/m1b/sqlite_rules/`). It did not help: −0.8 and −0.4 on
`bird_diag`, −3.8 and −2.5 on Spider dev (zero / rows3 + retry). The fine-tune stays
+13.2 / +10.8 above it on `bird_diag`. The rules were written after seeing `bird_diag`
errors, so if anything this check is biased in the prompt's favour.

## Findings

1. **No detectable BIRD gain for the 7B.** Δ is +0.4 with AWQ and exactly 0.0 with FP8. That
   is far under the pre-registered 5-point rule and not significant. With 250 questions the
   95% CI is about ±5.5, so a gain of a few points cannot be ruled out, but the difference
   to the 3B's gain is significant (+10.4, CI +3.3 to +17.5). The model does
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
4. **Why the 3B gained and the 7B did not.** The checks above support this reading. The 3B
   base makes many schema and dialect errors that 15k examples fix (base `no such column`
   share 30% → 14%, 003), and that skill transfers to any database. The 7B base already has
   it: its base validation loss (0.218) is close to the fine-tuned 3B (0.204), and on the 250
   training questions the 7B base (64.4) already matches the fine-tuned 3B (64.8). What the fine-tune adds on top
   is mostly gold-style writing, which lowers loss but not errors in results; the small
   real EX gain fades with distance from the training data (+7 seen, +2 unseen DBs, 0
   Mini-Dev). Not ruled out: more epochs or a larger rank might raise the 7B on its training
   questions, but that gain would have to survive the same fade to matter.
5. **Fair dev comparison for 003 finding 2.** With the same variant (rows3 + retry), the 3B
   fine-tune is 2.8 points below the 7B on `bird_diag` (p = 0.44). On the test set it was
   0.8 above (p = 0.89). Both say "level with the 7B within noise", not "better".

## Decision

- No 7B test-set run is needed to support a claim. The dev result is a clear null on BIRD.
  It stays a dev finding unless the test set is run (only with the user's approval).
- Headline model stays the 3B `m1b_mix`. The project result is "a 3B QLoRA lands within
  noise of the 4-bit 7B base, and a prompt cannot do the same", and "the same recipe does not
  lift the 7B" is the honest second half. For a 7B, this data is worth little; for a 3B it is
  the difference between below-7B and 7B-level BIRD accuracy at less than half the size.
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
- One seed per model. The BIRD validation part has only 3 databases (472 questions), so the
  in-distribution numbers are one more data point, not a precise estimate.
