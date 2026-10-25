# Run `m1_qwen7b_own__zero__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 74.1 |
| ex_col_tolerant | 77.9 |
| rate_correct | 74.1 |
| rate_wrong_result | 22.4 |
| rate_exec_error | 3.4 |
| rate_timeout | 0.1 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 305.9 |
| latency_p95_ms | 724.5 |
| mean_prompt_tokens | 387.4 |
| mean_completion_tokens | 33.8 |
| generation_s | 31.2 |

## Execution errors

| kind | count |
|---|---|
| no such column | 31 |
| ambiguous column | 2 |
| misuse of aggregate | 1 |
| syntax error | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| real_estate_properties | 4 | 50.0 |
| car_1 | 92 | 56.5 |
| student_transcripts_tracking | 78 | 66.7 |
| dog_kennels | 82 | 68.3 |
| course_teach | 30 | 70.0 |
| world_1 | 120 | 70.0 |
| wta_1 | 62 | 72.6 |
| flight_2 | 80 | 75.0 |
| battle_death | 16 | 75.0 |
| network_1 | 56 | 75.0 |
| tvshow | 62 | 82.3 |
| cre_Doc_Template_Mgt | 84 | 83.3 |
| voter_1 | 15 | 86.7 |
| museum_visit | 18 | 88.9 |
| pets_1 | 42 | 92.9 |
| singer | 30 | 96.7 |
| orchestra | 40 | 97.5 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |

