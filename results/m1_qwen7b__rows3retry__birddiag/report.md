# Run `m1_qwen7b__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 50.8 |
| ex_col_tolerant | 51.6 |
| rate_correct | 50.8 |
| rate_wrong_result | 44.4 |
| rate_exec_error | 4.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1061.3 |
| latency_p95_ms | 2443.0 |
| mean_prompt_tokens | 3006.1 |
| mean_completion_tokens | 74.5 |
| generation_s | 45.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 8 |
| syntax error | 2 |
| misuse of aggregate | 1 |
| other | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 26.7 |
| formula_1 | 66 | 43.9 |
| thrombosis_prediction | 50 | 46.0 |
| card_games | 52 | 55.8 |
| superhero | 52 | 76.9 |

