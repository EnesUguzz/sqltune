# Run `m1_m0_qlora__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 33.6 |
| ex_col_tolerant | 34.4 |
| rate_correct | 33.6 |
| rate_wrong_result | 40.0 |
| rate_exec_error | 26.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1202.9 |
| latency_p95_ms | 2985.3 |
| mean_prompt_tokens | 3449.4 |
| mean_completion_tokens | 80.3 |
| generation_s | 75.8 |

## Execution errors

| kind | count |
|---|---|
| no such column | 54 |
| syntax error | 10 |
| other | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 6.7 |
| thrombosis_prediction | 50 | 22.0 |
| formula_1 | 66 | 31.8 |
| card_games | 52 | 42.3 |
| superhero | 52 | 57.7 |

