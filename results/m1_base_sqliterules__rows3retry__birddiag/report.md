# Run `m1_base_sqliterules__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 37.6 |
| ex_col_tolerant | 38.0 |
| rate_correct | 37.6 |
| rate_wrong_result | 35.6 |
| rate_exec_error | 26.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1087.8 |
| latency_p95_ms | 2872.8 |
| mean_prompt_tokens | 3782.7 |
| mean_completion_tokens | 91.8 |
| generation_s | 62.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 63 |
| syntax error | 3 |
| no such table | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 28.0 |
| formula_1 | 66 | 37.9 |
| card_games | 52 | 38.5 |
| superhero | 52 | 63.5 |

