# Run `m1_m0_qlora__rows3retry__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 39.4 |
| ex_col_tolerant | 39.8 |
| rate_correct | 39.4 |
| rate_wrong_result | 45.0 |
| rate_exec_error | 15.7 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1113.6 |
| latency_p95_ms | 3023.8 |
| mean_prompt_tokens | 3132.9 |
| mean_completion_tokens | 81.1 |
| generation_s | 149.8 |

## Execution errors

| kind | count |
|---|---|
| no such column | 34 |
| other | 3 |
| syntax error | 1 |
| misuse of aggregate | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 6.2 |
| debit_card_specializing | 30 | 23.3 |
| toxicology | 40 | 35.0 |
| codebase_community | 48 | 37.5 |
| european_football_2 | 51 | 49.0 |
| student_club | 48 | 68.8 |

