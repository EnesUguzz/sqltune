# Run `m1_qwen7b_fp8__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 48.4 |
| ex_col_tolerant | 48.8 |
| rate_correct | 48.4 |
| rate_wrong_result | 42.4 |
| rate_exec_error | 9.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 902.1 |
| latency_p95_ms | 1386.8 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 64.5 |
| generation_s | 40.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 23 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 26.7 |
| formula_1 | 66 | 39.4 |
| card_games | 52 | 44.2 |
| thrombosis_prediction | 50 | 46.0 |
| superhero | 52 | 80.8 |

