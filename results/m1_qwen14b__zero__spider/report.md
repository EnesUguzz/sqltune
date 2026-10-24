# Run `m1_qwen14b__zero__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 77.4 |
| ex_col_tolerant | 82.1 |
| rate_correct | 77.4 |
| rate_wrong_result | 21.7 |
| rate_exec_error | 1.0 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 865.0 |
| latency_p95_ms | 1941.5 |
| mean_prompt_tokens | 387.4 |
| mean_completion_tokens | 32.1 |
| generation_s | 68.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 6 |
| ambiguous column | 4 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| car_1 | 92 | 64.1 |
| battle_death | 16 | 68.8 |
| student_transcripts_tracking | 78 | 70.5 |
| world_1 | 120 | 70.8 |
| real_estate_properties | 4 | 75.0 |
| wta_1 | 62 | 77.4 |
| dog_kennels | 82 | 79.3 |
| course_teach | 30 | 83.3 |
| flight_2 | 80 | 83.8 |
| cre_Doc_Template_Mgt | 84 | 86.9 |
| poker_player | 40 | 87.5 |
| network_1 | 56 | 87.5 |
| tvshow | 62 | 88.7 |
| concert_singer | 45 | 93.3 |
| voter_1 | 15 | 93.3 |
| museum_visit | 18 | 94.4 |
| pets_1 | 42 | 95.2 |
| singer | 30 | 96.7 |
| orchestra | 40 | 97.5 |
| employee_hire_evaluation | 38 | 100.0 |

