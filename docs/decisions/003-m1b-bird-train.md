# 003 — M1b: adding BIRD train to the 3B fine-tune

Date: 2026-10-24. Branch `m1b`. Plan: `docs/plans/2026-10-24-p2-m1b-bird-train-final.md`.
Data checks: `docs/data/dataset-findings.md`. Full table: `results/SUMMARY.md`.

## What was trained

One adapter, `m1b_mix`, trained from the base model (not on top of M0) with the M0 QLoRA
settings and 1 epoch:

- Spider train 6,540 + BIRD train 8,363 records (15k); 5% of each source held out as
  **whole databases** for validation (366 + 472 records).
- Leaky databases removed first: Spider `formula_1` / `soccer_1` (same data as two Mini-Dev
  databases), BIRD `world` / `airline` (same as Spider dev databases). 402 BIRD train
  questions whose gold SQL fails were skipped.
- Half of the records show 3 example rows per table, so both eval prompts (zero, rows3) are
  in the training distribution. `max_seq_length` 8192; 0 records dropped, 267 BIRD rows
  prompts fell back to the plain prompt.
- 93 min, 932 steps, peak 3.9 GiB VRAM, MLflow run `a33af0f0`.

**Validation loss:** 0.209 (step 200) → 0.210 → 0.205 → 0.200 → 0.201 (step 932). It is
nearly flat after the first fifth of the epoch: the format is learned early and later steps
gain little on unseen databases. This is a weak signal that more data of the *same* kind
would add only a few points; it does not say anything about different data (harder SQL,
other schemas). The BIRD validation part has only 3 databases, so it is noisy.

## Dev results (EX %, column-order tolerant)

| model / variant | bird_diag | Spider dev |
|---|---|---|
| base 3B, zero | 33.6 | 68.1 |
| base 3B, rows3 + retry | 38.4 | 73.0 |
| M0 adapter (Spider only), zero | 32.4 | 76.8 |
| M0 adapter, rows3 + retry | 34.4 | 80.3 |
| **m1b_mix, zero** | **46.0** | 76.2 |
| **m1b_mix, rows3 + retry** | **48.8** | 78.6 |
| 7B AWQ, retry | 48.4 | 80.1 |
| 14B AWQ, retry | 57.6 | 82.6 |

Paired exact McNemar (question level, same questions):

| comparison | set | fixed / broken | p |
|---|---|---|---|
| m1b_mix zero vs base zero | bird_diag | 46 / 15 | 8.8e-5 |
| m1b_mix rows3retry vs base rows3retry | bird_diag | 44 / 18 | 0.0013 |
| m1b_mix rows3retry vs 7B retry | bird_diag | 30 / 29 | 1.0 |
| m1b_mix vs M0, zero | Spider dev | | 0.62 |
| m1b_mix vs M0, rows3retry | Spider dev | | 0.14 |

Pre-registered rule: a difference under 5 points on `bird_diag` is not a gain. The 5-database
bootstrap CIs on `bird_diag` are indicative only.

## Findings

1. **BIRD train fixes the BIRD gap.** +12.4 (zero) and +10.4 (rows3 + retry) over the base on
   `bird_diag`, both well above the 5-point rule and significant. The Spider-only adapter
   was at or below the base here (M1a), so the gain comes from BIRD-style data (evidence
   hints, CASE/CAST/date SQL, wide schemas), not from fine-tuning as such.
2. **The 3B adapter reaches the 7B AWQ (4-bit) model on BIRD**, with a caveat on dev: the
   7B dev run used retry without example rows. Here m1b_mix with rows3 + retry scores 48.8 and
   7B with retry scores 48.4 (McNemar p = 1.0, 30 questions each way). The same-variant
   comparison is on the test set: 52.2 vs 51.4 (p = 0.89). The 3B is served in bf16, the 7B
   and 14B as AWQ 4-bit. 14B is still 9 points ahead. *Added 2026-10-25:* the same-variant
   dev run of the 7B (rows3 + retry) scores 51.6, 2.8 above m1b_mix (p = 0.44). So "level
   within noise" holds on both sets, and "better" holds on neither (see 004).
3. **Spider is not hurt.** m1b_mix vs the Spider-only M0 adapter: −0.6 (zero) and −1.7
   (rows3 + retry), neither significant. One adapter now covers both benchmarks.
4. **Schema errors drop most.** Share of `no such column` among `bird_diag` predictions:
   base 30% → m1b_mix 13.6%.
5. **Gains are larger on harder questions** (EX by BIRD difficulty, tolerant):

   | run | simple (70) | moderate (129) | challenging (51) |
   |---|---|---|---|
   | base, zero | 50.0 | 30.2 | 19.6 |
   | base, rows3 + retry | 55.7 | 33.3 | 27.5 |
   | M0 adapter, rows3 + retry | 52.9 | 31.0 | 17.6 |
   | m1b_mix, zero | 60.0 | 44.2 | 31.4 |
   | m1b_mix, rows3 + retry | 60.0 | 46.5 | 39.2 |
   | 7B AWQ, retry (no rows) | 64.3 | 44.2 | 37.3 |
   | 14B, retry | 71.4 | 51.2 | 54.9 |

   m1b_mix gains +13 on moderate and +12 on challenging over base (rows3 + retry), and
   beats the 7B retry run on both; the remaining gap to 7B and 14B is mostly on simple questions.

## What the remaining errors are

40 of the 128 failing `bird_diag` questions of m1b_mix (rows3 + retry), sampled with seed 0
(`random.Random(0).sample`), labelled by hand with one main cause each:

| cause | n | examples |
|---|---|---|
| schema linking: wrong table / column, or a column from the wrong table | 12 | bird-0091 (`HGB` is in Laboratory), bird-0368 (`format` is in legalities) |
| evidence hint not followed | 7 | bird-0187 / 0235 (evidence: larger dob = younger; model sorts the wrong way), bird-0383 (`translation IS NOT NULL` left out) |
| aggregation / logic | 6 | bird-0243 (total instead of max per driver), bird-0448 (`LIMIT 9, 2` vs top 10), bird-0454 (needs a window function) |
| gold suspect or ambiguous | 6 | bird-0115 (gold has an `AND`/`OR` precedence bug), bird-0125 / 0365 (model uses the value the evidence gives, gold uses another) |
| wrong output columns | 5 | bird-0109 (diagnosis missing), bird-0188 (name missing), bird-0465 (question asks two things) |
| value format / NULL handling | 3 | bird-0194 (text speed not cast to number), bird-0453 (NULL scores sort first) |
| SQL syntax | 1 | bird-0248 (`WHERE` before a second `JOIN`) |

So about 15% of sampled failures are not model errors (gold noise or ambiguity, a known
BIRD issue), and schema linking is still the largest group. That points to schema-side
help (pruning or a schema-linking step) or a larger model, more than to more of the same
training data.

## Final rows for the test sets

base 3B, M0 adapter, m1b_mix, 7B AWQ, 14B AWQ; each with `zero` and `rows3 + retry`. Run
once on Spider test (2,147) and `bird_holdout` (250, 6 databases) — see below.

## Test results (run once, 2026-10-24)

EX %, column-order tolerant (strict in brackets). `bird_holdout`: 250 questions, 6 databases
never used for any decision. Spider test: 2,147 questions. No transport errors in any run.

| model | variant | bird_holdout | Spider test |
|---|---|---|---|
| base 3B | zero | 40.6 (40.6) | 71.7 (67.5) |
| base 3B | rows3 + retry | 46.2 (46.2) | 76.6 (72.3) |
| M0 adapter (Spider only) | zero | 36.1 (36.1) | 77.7 (74.1) |
| M0 adapter | rows3 + retry | 39.8 (39.4) | 80.2 (75.6) |
| **m1b_mix (3B)** | zero | **50.6** (50.6) | 78.6 (74.3) |
| **m1b_mix (3B)** | rows3 + retry | **52.2** (52.2) | 80.9 (76.4) |
| 7B AWQ | zero | 46.6 (46.2) | 78.7 (74.0) |
| 7B AWQ | rows3 + retry | 51.4 (51.0) | 82.6 (77.4) |
| 14B AWQ | zero | 54.2 (53.8) | 82.5 (77.7) |
| 14B AWQ | rows3 + retry | 60.2 (59.8) | 85.8 (80.9) |

McNemar, m1b_mix against:

| set | comparison | Δ | fixed / broken | p |
|---|---|---|---|---|
| bird_holdout | base, zero | +10.0 | 36 / 11 | 3.5e-4 |
| bird_holdout | base, rows3 + retry | +6.0 | 27 / 12 | 0.024 |
| bird_holdout | M0, rows3 + retry | +12.4 | 42 / 11 | 2.2e-5 |
| bird_holdout | 7B, rows3 + retry | +0.8 | 25 / 23 | 0.89 |
| bird_holdout | 14B, rows3 + retry | −8.0 | 19 / 39 | 0.012 |
| Spider test | base, zero | +6.9 | 242 / 94 | 3.5e-16 |
| Spider test | M0, rows3 + retry | +0.7 | 93 / 77 | 0.25 |
| Spider test | 7B, rows3 + retry | −1.7 | 130 / 167 | 0.037 |

**Dev vs test.** The dev conclusions hold on the untouched sets:

- BIRD train helps on BIRD: +10.0 zero-shot over base, and +6.0 with rows + retry. That is
  smaller than on `bird_diag` (+10.4), but still above the 5-point rule and significant.
- The Spider-only M0 adapter is again below the base on BIRD (−4.4 zero-shot), so that finding
  was not caused by the `formula_1` overlap in `bird_diag`.
- m1b_mix ties 7B on BIRD (+0.8, p = 0.89).
- On Spider, m1b_mix is 1.7 points below 7B (p = 0.04) and level with M0.
- 14B stays clearly ahead on both sets.

`bird_holdout` is easier than `bird_diag` for every model (base zero 40.6 vs 33.6), so
compare models within a set, not across sets.

## Caveats

- The M0 and M1a adapters saw `formula_1` (a `bird_diag` database) and `soccer_1` (same
  data as a `bird_holdout` database) in training. They still did not beat the base on BIRD,
  so the M1a conclusion holds; m1b_mix had both removed.
- `bird_holdout` is a 6-database subset of Mini-Dev; numbers are not directly comparable to
  published Mini-Dev or BIRD dev scores.
- Spider test EX is measured with this repo's harness (not the official test-suite).
- The 3B models are served in bf16 (base + LoRA); 7B and 14B are the official AWQ 4-bit
  checkpoints, so "3B ≈ 7B" compares a full-precision 3B with a 4-bit 7B.
- Qwen2.5-Coder may have seen Spider / BIRD during pre-training; absolute scores may be
  inflated, deltas against the same base are still valid.

## Next (decided 2026-10-24)

- **7B fine-tune** with the same data and settings (`m1b/train/mix7b.yaml`), served as
  merge → AWQ 4-bit. The base 7B is quantized with the same script and calibration set
  (`scripts/merge_awq.py` without `--adapter`), so the two differ only by the fine-tune. Question:
  does the BIRD-train gain on top of a 7B base close the gap to 14B? **Answer in
  [004](004-m1b-7b-finetune.md): no. The 7B gains 0 on BIRD dev, also when served in FP8.**
- Haiku (one-shot, `claude -p`) as a closed-model reference, and a Gradio demo.
