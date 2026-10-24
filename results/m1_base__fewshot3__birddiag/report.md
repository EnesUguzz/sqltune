# Run `m1_base__fewshot3__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 32.0 |
| ex_col_tolerant | 32.4 |
| rate_correct | 32.0 |
| rate_wrong_result | 27.2 |
| rate_exec_error | 40.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 3492.0 |
| latency_p95_ms | 5700.8 |
| mean_prompt_tokens | 2405.5 |
| mean_completion_tokens | 64.2 |
| generation_s | 94.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 86 |
| other | 12 |
| misuse of aggregate | 2 |
| no such table | 1 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 13.3 |
| thrombosis_prediction | 50 | 22.0 |
| formula_1 | 66 | 31.8 |
| card_games | 52 | 32.7 |
| superhero | 52 | 53.8 |

