# Run `m1_qwen7b__rows3retry__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 51.0 |
| ex_col_tolerant | 51.4 |
| rate_correct | 51.0 |
| rate_wrong_result | 41.8 |
| rate_exec_error | 7.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1005.9 |
| latency_p95_ms | 2147.5 |
| mean_prompt_tokens | 2796.3 |
| mean_completion_tokens | 72.7 |
| generation_s | 143.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 11 |
| misuse of aggregate | 3 |
| no such table | 2 |
| other | 1 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| toxicology | 40 | 32.5 |
| financial | 32 | 37.5 |
| codebase_community | 48 | 41.7 |
| debit_card_specializing | 30 | 53.3 |
| european_football_2 | 51 | 62.7 |
| student_club | 48 | 72.9 |

