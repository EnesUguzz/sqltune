# Run `m1_m1b_mix__rows3retry__birdholdout`

| metric | value |
|---|---|
| n | 250 |
| n_gold_error | 1 |
| n_transport_error | 0 |
| ex | 52.2 |
| ex_col_tolerant | 52.2 |
| rate_correct | 52.2 |
| rate_wrong_result | 40.6 |
| rate_exec_error | 7.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 1108.4 |
| latency_p95_ms | 2447.2 |
| mean_prompt_tokens | 2655.7 |
| mean_completion_tokens | 71.9 |
| generation_s | 145.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 17 |
| other | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| financial | 32 | 6.2 |
| toxicology | 40 | 40.0 |
| debit_card_specializing | 30 | 46.7 |
| codebase_community | 48 | 58.3 |
| european_football_2 | 51 | 66.7 |
| student_club | 48 | 75.0 |

