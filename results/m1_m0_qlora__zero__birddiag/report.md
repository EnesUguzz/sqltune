# Run `m1_m0_qlora__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 31.2 |
| ex_col_tolerant | 32.4 |
| rate_correct | 31.2 |
| rate_wrong_result | 32.4 |
| rate_exec_error | 36.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 825.8 |
| latency_p95_ms | 1343.9 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 60.3 |
| generation_s | 45.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 72 |
| other | 10 |
| syntax error | 9 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 6.7 |
| thrombosis_prediction | 50 | 18.0 |
| formula_1 | 66 | 30.3 |
| card_games | 52 | 38.5 |
| superhero | 52 | 57.7 |

