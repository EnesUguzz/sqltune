# Run `m1_base__zero__birdval`

| metric | value |
|---|---|
| n | 472 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 35.0 |
| ex_col_tolerant | 36.0 |
| rate_correct | 35.0 |
| rate_wrong_result | 39.4 |
| rate_exec_error | 25.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.4 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 771.4 |
| latency_p95_ms | 1285.3 |
| mean_prompt_tokens | 1473.1 |
| mean_completion_tokens | 60.9 |
| generation_s | 39.7 |

## Execution errors

| kind | count |
|---|---|
| no such column | 94 |
| no such table | 9 |
| syntax error | 8 |
| misuse of aggregate | 5 |
| other | 3 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| hockey | 206 | 23.8 |
| regional_sales | 164 | 38.4 |
| sales | 102 | 56.9 |

