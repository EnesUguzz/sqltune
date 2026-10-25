# Run `m1_qwen7b_own__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 50.0 |
| ex_col_tolerant | 51.6 |
| rate_correct | 50.0 |
| rate_wrong_result | 44.4 |
| rate_exec_error | 5.6 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1046.1 |
| latency_p95_ms | 2264.2 |
| mean_prompt_tokens | 2936.0 |
| mean_completion_tokens | 71.7 |
| generation_s | 45.7 |

## Execution errors

| kind | count |
|---|---|
| no such column | 13 |
| other | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 23.3 |
| formula_1 | 66 | 43.9 |
| thrombosis_prediction | 50 | 44.0 |
| card_games | 52 | 51.9 |
| superhero | 52 | 84.6 |

