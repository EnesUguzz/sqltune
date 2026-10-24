# Run `m1_lc_f0125__zero__birddiag`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 30.1 |
| ex_col_tolerant | 30.5 |
| rate_correct | 30.1 |
| rate_wrong_result | 33.7 |
| rate_exec_error | 36.1 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1928.5 |
| latency_p95_ms | 3629.9 |
| mean_prompt_tokens | 956.4 |
| mean_completion_tokens | 58.4 |
| generation_s | 64.4 |

## Execution errors

| kind | count |
|---|---|
| no such column | 72 |
| other | 11 |
| ambiguous column | 3 |
| syntax error | 2 |
| no such table | 1 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| california_schools | 30 | 13.3 |
| thrombosis_prediction | 50 | 22.0 |
| formula_1 | 66 | 28.8 |
| card_games | 51 | 31.4 |
| superhero | 52 | 50.0 |

