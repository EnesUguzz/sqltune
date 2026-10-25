# Run `m1_qwen7b_fp8__zero__birdval`

| metric | value |
|---|---|
| n | 472 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 55.9 |
| ex_col_tolerant | 56.4 |
| rate_correct | 55.9 |
| rate_wrong_result | 36.0 |
| rate_exec_error | 7.2 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.8 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 866.1 |
| latency_p95_ms | 1572.0 |
| mean_prompt_tokens | 1473.1 |
| mean_completion_tokens | 59.3 |
| generation_s | 41.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 27 |
| misuse of aggregate | 4 |
| ambiguous column | 2 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| hockey | 206 | 43.7 |
| regional_sales | 164 | 64.6 |
| sales | 102 | 68.6 |

