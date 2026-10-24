# Run `m1_lc_f025__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 28.5 |
| ex_col_tolerant | 28.9 |
| rate_correct | 28.5 |
| rate_wrong_result | 32.5 |
| rate_exec_error | 39.0 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1356.7 |
| latency_p95_ms | 2579.0 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 58.4 |
| generation_s | 55.4 |

## Execution errors

| kind | count |
|---|---|
| no such column | 78 |
| other | 11 |
| syntax error | 7 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 20.0 |
| card_games | 51 | 29.4 |
| formula_1 | 66 | 31.8 |
| superhero | 52 | 44.2 |

