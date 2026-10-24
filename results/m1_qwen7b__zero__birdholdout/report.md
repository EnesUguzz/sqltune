# Run `m1_qwen7b__zero__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 46.2 |
| ex_col_tolerant | 46.6 |
| rate_correct | 46.2 |
| rate_wrong_result | 43.0 |
| rate_exec_error | 10.8 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 767.3 |
| latency_p95_ms | 1352.5 |
| mean_prompt_tokens | 940.3 |
| mean_completion_tokens | 64.4 |
| generation_s | 134.9 |

## Execution errors

| kind | count |
|---|---|
| no such column | 18 |
| misuse of aggregate | 3 |
| no such table | 2 |
| ambiguous column | 2 |
| other | 1 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| toxicology | 40 | 25.0 |
| financial | 32 | 25.0 |
| codebase_community | 48 | 37.5 |
| debit_card_specializing | 30 | 46.7 |
| european_football_2 | 51 | 62.7 |
| student_club | 48 | 70.8 |

