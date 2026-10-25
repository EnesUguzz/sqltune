# Run `m1_qwen7b_own__rows3retry__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 77.9 |
| ex_col_tolerant | 81.6 |
| rate_correct | 77.9 |
| rate_wrong_result | 20.8 |
| rate_exec_error | 1.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 343.1 |
| latency_p95_ms | 938.4 |
| mean_prompt_tokens | 954.4 |
| mean_completion_tokens | 34.7 |
| generation_s | 32.6 |

## Execution errors

| kind | count |
|---|---|
| no such column | 13 |
| ambiguous column | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| car_1 | 92 | 69.6 |
| dog_kennels | 82 | 72.0 |
| world_1 | 120 | 72.5 |
| wta_1 | 62 | 72.6 |
| student_transcripts_tracking | 78 | 73.1 |
| battle_death | 16 | 75.0 |
| real_estate_properties | 4 | 75.0 |
| network_1 | 56 | 78.6 |
| flight_2 | 80 | 78.8 |
| course_teach | 30 | 83.3 |
| cre_Doc_Template_Mgt | 84 | 85.7 |
| voter_1 | 15 | 86.7 |
| museum_visit | 18 | 88.9 |
| tvshow | 62 | 90.3 |
| orchestra | 40 | 92.5 |
| pets_1 | 42 | 92.9 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |
| singer | 30 | 100.0 |

