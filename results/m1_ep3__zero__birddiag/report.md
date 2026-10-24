# Run `m1_ep3__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 29.7 |
| ex_col_tolerant | 30.9 |
| rate_correct | 29.7 |
| rate_wrong_result | 32.5 |
| rate_exec_error | 37.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1551.5 |
| latency_p95_ms | 3045.9 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 57.7 |
| generation_s | 60.9 |

## Execution errors

| kind | count |
|---|---|
| no such column | 69 |
| other | 13 |
| syntax error | 9 |
| ambiguous column | 1 |
| misuse of aggregate | 1 |
| no such table | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 3.3 |
| thrombosis_prediction | 50 | 16.0 |
| formula_1 | 66 | 31.8 |
| card_games | 51 | 35.3 |
| superhero | 52 | 55.8 |

