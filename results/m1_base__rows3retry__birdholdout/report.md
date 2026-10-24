# Run `m1_base__rows3retry__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 46.2 |
| ex_col_tolerant | 46.2 |
| rate_correct | 46.2 |
| rate_wrong_result | 34.9 |
| rate_exec_error | 18.9 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1019.7 |
| latency_p95_ms | 2552.5 |
| mean_prompt_tokens | 3149.6 |
| mean_completion_tokens | 80.9 |
| generation_s | 143.9 |

## Execution errors

| kind | count |
|---|---|
| no such column | 40 |
| other | 3 |
| misuse of aggregate | 2 |
| no such table | 1 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 12.5 |
| toxicology | 40 | 30.0 |
| debit_card_specializing | 30 | 40.0 |
| codebase_community | 48 | 50.0 |
| european_football_2 | 51 | 60.8 |
| student_club | 48 | 66.7 |

