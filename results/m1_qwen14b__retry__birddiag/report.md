# Run `m1_qwen14b__retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 56.8 |
| ex_col_tolerant | 57.6 |
| rate_correct | 56.8 |
| rate_wrong_result | 42.8 |
| rate_exec_error | 0.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 2264.3 |
| latency_p95_ms | 53300.9 |
| mean_prompt_tokens | 1010.9 |
| mean_completion_tokens | 66.7 |
| generation_s | 126.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 36.7 |
| thrombosis_prediction | 50 | 46.0 |
| card_games | 52 | 53.8 |
| formula_1 | 66 | 57.6 |
| superhero | 52 | 84.6 |

