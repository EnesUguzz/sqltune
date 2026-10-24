# Run `m1_lora16__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 32.1 |
| ex_col_tolerant | 32.9 |
| rate_correct | 32.1 |
| rate_wrong_result | 37.8 |
| rate_exec_error | 30.1 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1574.0 |
| latency_p95_ms | 2528.3 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 59.4 |
| generation_s | 59.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 56 |
| other | 12 |
| syntax error | 6 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 10.0 |
| thrombosis_prediction | 50 | 20.0 |
| formula_1 | 66 | 33.3 |
| card_games | 51 | 35.3 |
| superhero | 52 | 55.8 |

