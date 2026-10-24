# Run `m1_seed3__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 30.4 |
| ex_col_tolerant | 31.2 |
| rate_correct | 30.4 |
| rate_wrong_result | 34.8 |
| rate_exec_error | 34.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 892.2 |
| latency_p95_ms | 1510.7 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 61.6 |
| generation_s | 45.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 64 |
| other | 14 |
| syntax error | 6 |
| no such table | 2 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 6.7 |
| thrombosis_prediction | 50 | 20.0 |
| formula_1 | 66 | 30.3 |
| card_games | 52 | 36.5 |
| superhero | 52 | 51.9 |

