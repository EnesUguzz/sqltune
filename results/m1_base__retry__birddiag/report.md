# Run `m1_base__retry__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 35.6 |
| ex_col_tolerant | 36.4 |
| rate_correct | 35.6 |
| rate_wrong_result | 36.4 |
| rate_exec_error | 28.0 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1055.5 |
| latency_p95_ms | 2738.5 |
| mean_prompt_tokens | 1339.6 |
| mean_completion_tokens | 85.4 |
| generation_s | 52.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 66 |
| syntax error | 2 |
| other | 1 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 26.0 |
| formula_1 | 66 | 31.8 |
| card_games | 52 | 38.5 |
| superhero | 52 | 65.4 |

