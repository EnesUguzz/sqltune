# Run `m1_m0_qlora__retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 32.0 |
| ex_col_tolerant | 33.2 |
| rate_correct | 32.0 |
| rate_wrong_result | 40.8 |
| rate_exec_error | 27.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1458.1 |
| latency_p95_ms | 6798.1 |
| mean_prompt_tokens | 1315.3 |
| mean_completion_tokens | 85.9 |
| generation_s | 96.7 |

## Execution errors

| kind | count |
|---|---|
| no such column | 54 |
| syntax error | 8 |
| other | 5 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 6.7 |
| thrombosis_prediction | 50 | 20.0 |
| formula_1 | 66 | 31.8 |
| card_games | 52 | 38.5 |
| superhero | 52 | 57.7 |

