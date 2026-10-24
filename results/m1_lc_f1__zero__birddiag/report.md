# Run `m1_lc_f1__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 31.6 |
| ex_col_tolerant | 32.4 |
| rate_correct | 31.6 |
| rate_wrong_result | 34.8 |
| rate_exec_error | 33.6 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1563.6 |
| latency_p95_ms | 2456.3 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 57.5 |
| generation_s | 57.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 65 |
| other | 10 |
| syntax error | 8 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 6.7 |
| thrombosis_prediction | 50 | 18.0 |
| formula_1 | 66 | 30.3 |
| card_games | 52 | 38.5 |
| superhero | 52 | 57.7 |

