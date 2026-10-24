# Run `m1_base__zero__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 40.6 |
| ex_col_tolerant | 40.6 |
| rate_correct | 40.6 |
| rate_wrong_result | 30.1 |
| rate_exec_error | 29.3 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 762.0 |
| latency_p95_ms | 1316.0 |
| mean_prompt_tokens | 940.3 |
| mean_completion_tokens | 65.7 |
| generation_s | 134.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 62 |
| other | 5 |
| misuse of aggregate | 4 |
| no such table | 1 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 9.4 |
| toxicology | 40 | 25.0 |
| debit_card_specializing | 30 | 40.0 |
| codebase_community | 48 | 41.7 |
| student_club | 48 | 56.2 |
| european_football_2 | 51 | 56.9 |

