# Run `m1_base__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 33.2 |
| ex_col_tolerant | 33.6 |
| rate_correct | 33.2 |
| rate_wrong_result | 31.6 |
| rate_exec_error | 35.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 775.0 |
| latency_p95_ms | 1288.5 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 61.0 |
| generation_s | 37.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 75 |
| other | 10 |
| syntax error | 2 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 24.0 |
| formula_1 | 66 | 28.8 |
| card_games | 52 | 38.5 |
| superhero | 52 | 57.7 |

