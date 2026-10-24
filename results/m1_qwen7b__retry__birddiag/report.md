# Run `m1_qwen7b__retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 47.6 |
| ex_col_tolerant | 48.4 |
| rate_correct | 47.6 |
| rate_wrong_result | 46.0 |
| rate_exec_error | 6.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 938.2 |
| latency_p95_ms | 2209.8 |
| mean_prompt_tokens | 1115.3 |
| mean_completion_tokens | 76.7 |
| generation_s | 46.7 |

## Execution errors

| kind | count |
|---|---|
| no such column | 16 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 26.7 |
| formula_1 | 66 | 37.9 |
| thrombosis_prediction | 50 | 42.0 |
| card_games | 52 | 50.0 |
| superhero | 52 | 78.8 |

