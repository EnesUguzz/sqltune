# Run `m1_base__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 37.8 |
| ex_col_tolerant | 38.6 |
| rate_correct | 37.8 |
| rate_wrong_result | 34.9 |
| rate_exec_error | 27.3 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1713.3 |
| latency_p95_ms | 4620.1 |
| mean_prompt_tokens | 3629.8 |
| mean_completion_tokens | 90.3 |
| generation_s | 80.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 62 |
| syntax error | 5 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 28.0 |
| formula_1 | 66 | 36.4 |
| card_games | 51 | 43.1 |
| superhero | 52 | 63.5 |

