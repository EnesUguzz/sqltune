# Run `m1_qwen7b_own__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 45.2 |
| ex_col_tolerant | 46.4 |
| rate_correct | 45.2 |
| rate_wrong_result | 42.0 |
| rate_exec_error | 12.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 799.7 |
| latency_p95_ms | 1297.8 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 65.2 |
| generation_s | 40.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 31 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 16.7 |
| formula_1 | 66 | 39.4 |
| thrombosis_prediction | 50 | 46.0 |
| card_games | 52 | 48.1 |
| superhero | 52 | 71.2 |

