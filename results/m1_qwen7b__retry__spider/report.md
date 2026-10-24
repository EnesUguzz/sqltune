# Run `m1_qwen7b__retry__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 76.0 |
| ex_col_tolerant | 80.1 |
| rate_correct | 76.0 |
| rate_wrong_result | 22.0 |
| rate_exec_error | 2.0 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 343.0 |
| latency_p95_ms | 895.7 |
| mean_prompt_tokens | 405.7 |
| mean_completion_tokens | 35.7 |
| generation_s | 32.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 17 |
| misuse of aggregate | 2 |
| ambiguous column | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| real_estate_properties | 4 | 50.0 |
| car_1 | 92 | 60.9 |
| world_1 | 120 | 69.2 |
| student_transcripts_tracking | 78 | 71.8 |
| wta_1 | 62 | 72.6 |
| dog_kennels | 82 | 74.4 |
| battle_death | 16 | 75.0 |
| network_1 | 56 | 78.6 |
| flight_2 | 80 | 78.8 |
| cre_Doc_Template_Mgt | 84 | 83.3 |
| course_teach | 30 | 83.3 |
| tvshow | 62 | 85.5 |
| voter_1 | 15 | 86.7 |
| orchestra | 40 | 92.5 |
| museum_visit | 18 | 94.4 |
| pets_1 | 42 | 95.2 |
| singer | 30 | 96.7 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |

