# Run `m1_qwen7b__rows3retry__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 78.1 |
| ex_col_tolerant | 82.4 |
| rate_correct | 78.1 |
| rate_wrong_result | 20.5 |
| rate_exec_error | 1.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 341.9 |
| latency_p95_ms | 964.0 |
| mean_prompt_tokens | 956.6 |
| mean_completion_tokens | 34.5 |
| generation_s | 32.0 |

## Execution errors

| kind | count |
|---|---|
| no such column | 12 |
| ambiguous column | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| car_1 | 92 | 67.4 |
| battle_death | 16 | 68.8 |
| dog_kennels | 82 | 72.0 |
| course_teach | 30 | 73.3 |
| world_1 | 120 | 75.0 |
| real_estate_properties | 4 | 75.0 |
| student_transcripts_tracking | 78 | 75.6 |
| flight_2 | 80 | 77.5 |
| wta_1 | 62 | 79.0 |
| network_1 | 56 | 83.9 |
| cre_Doc_Template_Mgt | 84 | 86.9 |
| orchestra | 40 | 90.0 |
| pets_1 | 42 | 90.5 |
| voter_1 | 15 | 93.3 |
| tvshow | 62 | 93.5 |
| museum_visit | 18 | 94.4 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |
| singer | 30 | 100.0 |

