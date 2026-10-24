# Run `m1_qwen7b__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 44.8 |
| ex_col_tolerant | 45.6 |
| rate_correct | 44.8 |
| rate_wrong_result | 40.4 |
| rate_exec_error | 14.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 845.7 |
| latency_p95_ms | 1358.1 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 64.9 |
| generation_s | 41.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 35 |
| syntax error | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 20.0 |
| formula_1 | 66 | 36.4 |
| thrombosis_prediction | 50 | 42.0 |
| card_games | 52 | 48.1 |
| superhero | 52 | 73.1 |

