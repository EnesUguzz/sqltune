# Run `m1_m1b_mix__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 48.4 |
| ex_col_tolerant | 48.8 |
| rate_correct | 48.4 |
| rate_wrong_result | 40.4 |
| rate_exec_error | 11.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1114.4 |
| latency_p95_ms | 2355.8 |
| mean_prompt_tokens | 2939.9 |
| mean_completion_tokens | 69.9 |
| generation_s | 46.5 |

## Execution errors

| kind | count |
|---|---|
| no such column | 27 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 26.7 |
| thrombosis_prediction | 50 | 40.0 |
| formula_1 | 66 | 42.4 |
| card_games | 52 | 53.8 |
| superhero | 52 | 73.1 |

