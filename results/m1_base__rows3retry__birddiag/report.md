# Run `m1_base__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 37.6 |
| ex_col_tolerant | 38.4 |
| rate_correct | 37.6 |
| rate_wrong_result | 34.8 |
| rate_exec_error | 27.6 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1065.8 |
| latency_p95_ms | 2754.3 |
| mean_prompt_tokens | 3647.6 |
| mean_completion_tokens | 90.9 |
| generation_s | 60.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 61 |
| syntax error | 7 |
| no such table | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 28.0 |
| formula_1 | 66 | 37.9 |
| card_games | 52 | 40.4 |
| superhero | 52 | 63.5 |

