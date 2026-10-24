# Run `m1_seed2__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 31.6 |
| ex_col_tolerant | 32.0 |
| rate_correct | 31.6 |
| rate_wrong_result | 33.2 |
| rate_exec_error | 35.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 922.9 |
| latency_p95_ms | 1419.3 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 57.1 |
| generation_s | 42.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 63 |
| other | 12 |
| syntax error | 8 |
| ambiguous column | 4 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 16.0 |
| formula_1 | 66 | 30.3 |
| card_games | 52 | 40.4 |
| superhero | 52 | 53.8 |

