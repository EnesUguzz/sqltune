# Run `m1_qwen7b_fp8__rows3retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 52.8 |
| ex_col_tolerant | 53.2 |
| rate_correct | 52.8 |
| rate_wrong_result | 43.2 |
| rate_exec_error | 4.0 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1066.3 |
| latency_p95_ms | 2527.9 |
| mean_prompt_tokens | 2935.3 |
| mean_completion_tokens | 70.8 |
| generation_s | 46.4 |

## Execution errors

| kind | count |
|---|---|
| no such column | 9 |
| other | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 26.7 |
| formula_1 | 66 | 43.9 |
| thrombosis_prediction | 50 | 44.0 |
| card_games | 52 | 55.8 |
| superhero | 52 | 86.5 |

