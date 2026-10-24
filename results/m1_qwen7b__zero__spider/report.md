# Run `m1_qwen7b__zero__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 74.7 |
| ex_col_tolerant | 78.7 |
| rate_correct | 74.7 |
| rate_wrong_result | 21.5 |
| rate_exec_error | 3.9 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 299.5 |
| latency_p95_ms | 718.2 |
| mean_prompt_tokens | 387.4 |
| mean_completion_tokens | 33.6 |
| generation_s | 26.3 |

## Execution errors

| kind | count |
|---|---|
| no such column | 34 |
| ambiguous column | 4 |
| misuse of aggregate | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| real_estate_properties | 4 | 50.0 |
| car_1 | 92 | 59.8 |
| world_1 | 120 | 69.2 |
| course_teach | 30 | 70.0 |
| student_transcripts_tracking | 78 | 70.5 |
| wta_1 | 62 | 72.6 |
| dog_kennels | 82 | 73.2 |
| flight_2 | 80 | 73.8 |
| battle_death | 16 | 75.0 |
| network_1 | 56 | 78.6 |
| cre_Doc_Template_Mgt | 84 | 82.1 |
| tvshow | 62 | 83.9 |
| voter_1 | 15 | 86.7 |
| museum_visit | 18 | 88.9 |
| orchestra | 40 | 92.5 |
| pets_1 | 42 | 95.2 |
| singer | 30 | 96.7 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |

