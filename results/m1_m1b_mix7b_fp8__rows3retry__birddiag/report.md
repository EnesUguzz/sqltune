# Run `m1_m1b_mix7b_fp8__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 52.4 |
| ex_col_tolerant | 53.2 |
| rate_correct | 52.4 |
| rate_wrong_result | 44.0 |
| rate_exec_error | 3.6 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1097.6 |
| latency_p95_ms | 2023.6 |
| mean_prompt_tokens | 2784.0 |
| mean_completion_tokens | 66.7 |
| generation_s | 53.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 6 |
| syntax error | 3 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 30.0 |
| formula_1 | 66 | 48.5 |
| thrombosis_prediction | 50 | 50.0 |
| card_games | 52 | 51.9 |
| superhero | 52 | 76.9 |

