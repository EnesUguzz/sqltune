# Run `m1_qwen14b__rows3retry__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 59.8 |
| ex_col_tolerant | 60.2 |
| rate_correct | 59.8 |
| rate_wrong_result | 39.0 |
| rate_exec_error | 1.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 3212.7 |
| latency_p95_ms | 49919.5 |
| mean_prompt_tokens | 2574.3 |
| mean_completion_tokens | 66.0 |
| generation_s | 235.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 1 |
| ambiguous column | 1 |
| no such table | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| codebase_community | 48 | 45.8 |
| financial | 32 | 46.9 |
| toxicology | 40 | 52.5 |
| debit_card_specializing | 30 | 56.7 |
| european_football_2 | 51 | 70.6 |
| student_club | 48 | 81.2 |

