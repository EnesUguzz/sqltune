# Run `m1_qwen14b__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 54.8 |
| ex_col_tolerant | 55.6 |
| rate_correct | 54.8 |
| rate_wrong_result | 40.4 |
| rate_exec_error | 4.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 2242.8 |
| latency_p95_ms | 3758.0 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 63.4 |
| generation_s | 64.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 9 |
| other | 2 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 33.3 |
| thrombosis_prediction | 50 | 42.0 |
| card_games | 52 | 53.8 |
| formula_1 | 66 | 54.5 |
| superhero | 52 | 84.6 |

