# Run `m1_qwen14b__zero__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 53.8 |
| ex_col_tolerant | 54.2 |
| rate_correct | 53.8 |
| rate_wrong_result | 41.8 |
| rate_exec_error | 4.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 2834.6 |
| latency_p95_ms | 6734.6 |
| mean_prompt_tokens | 940.3 |
| mean_completion_tokens | 63.3 |
| generation_s | 175.8 |

## Execution errors

| kind | count |
|---|---|
| no such column | 8 |
| misuse of aggregate | 1 |
| ambiguous column | 1 |
| no such table | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 37.5 |
| codebase_community | 48 | 41.7 |
| toxicology | 40 | 45.0 |
| debit_card_specializing | 30 | 46.7 |
| european_football_2 | 51 | 70.6 |
| student_club | 48 | 72.9 |

