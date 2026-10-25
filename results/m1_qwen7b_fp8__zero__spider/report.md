# Run `m1_qwen7b_fp8__zero__spider`

| metric | value |
|---|---|
| n | 1034 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 74.3 |
| ex_col_tolerant | 78.1 |
| rate_correct | 74.3 |
| rate_wrong_result | 22.3 |
| rate_exec_error | 3.4 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 374.8 |
| latency_p95_ms | 886.8 |
| mean_prompt_tokens | 387.4 |
| mean_completion_tokens | 33.7 |
| generation_s | 32.1 |

## Execution errors

| kind | count |
|---|---|
| no such column | 27 |
| ambiguous column | 6 |
| misuse of aggregate | 2 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| real_estate_properties | 4 | 50.0 |
| car_1 | 92 | 58.7 |
| dog_kennels | 82 | 63.4 |
| student_transcripts_tracking | 78 | 67.9 |
| world_1 | 120 | 70.0 |
| wta_1 | 62 | 71.0 |
| course_teach | 30 | 73.3 |
| battle_death | 16 | 75.0 |
| network_1 | 56 | 75.0 |
| flight_2 | 80 | 77.5 |
| cre_Doc_Template_Mgt | 84 | 82.1 |
| tvshow | 62 | 85.5 |
| museum_visit | 18 | 88.9 |
| pets_1 | 42 | 92.9 |
| voter_1 | 15 | 93.3 |
| singer | 30 | 96.7 |
| orchestra | 40 | 97.5 |
| concert_singer | 45 | 97.8 |
| employee_hire_evaluation | 38 | 100.0 |
| poker_player | 40 | 100.0 |

