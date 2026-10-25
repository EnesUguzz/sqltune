# Run `m1_qwen7b_fp8__rows3retry__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 78.9 |
| ex_col_tolerant | 83.0 |
| rate_correct | 78.9 |
| rate_wrong_result | 20.2 |
| rate_exec_error | 0.9 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 397.5 |
| latency_p95_ms | 1072.9 |
| mean_prompt_tokens | 955.8 |
| mean_completion_tokens | 34.2 |
| generation_s | 36.8 |

## Execution errors

| kind | count |
|---|---|
| no such column | 7 |
| misuse of aggregate | 1 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| dog_kennels | 82 | 69.5 |
| car_1 | 92 | 72.8 |
| network_1 | 56 | 73.2 |
| wta_1 | 62 | 74.2 |
| student_transcripts_tracking | 78 | 74.4 |
| battle_death | 16 | 75.0 |
| world_1 | 120 | 75.0 |
| real_estate_properties | 4 | 75.0 |
| cre_Doc_Template_Mgt | 84 | 86.9 |
| pets_1 | 42 | 88.1 |
| flight_2 | 80 | 88.8 |
| museum_visit | 18 | 88.9 |
| course_teach | 30 | 90.0 |
| tvshow | 62 | 90.3 |
| voter_1 | 15 | 93.3 |
| orchestra | 40 | 95.0 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |
| singer | 30 | 100.0 |

