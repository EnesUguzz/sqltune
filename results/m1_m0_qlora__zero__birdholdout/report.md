# Run `m1_m0_qlora__zero__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 36.1 |
| ex_col_tolerant | 36.1 |
| rate_correct | 36.1 |
| rate_wrong_result | 43.8 |
| rate_exec_error | 20.1 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 884.9 |
| latency_p95_ms | 1656.4 |
| mean_prompt_tokens | 940.3 |
| mean_completion_tokens | 63.5 |
| generation_s | 137.7 |

## Execution errors

| kind | count |
|---|---|
| no such column | 41 |
| other | 6 |
| misuse of aggregate | 2 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 12.5 |
| debit_card_specializing | 30 | 13.3 |
| toxicology | 40 | 30.0 |
| codebase_community | 48 | 35.4 |
| european_football_2 | 51 | 45.1 |
| student_club | 48 | 62.5 |

