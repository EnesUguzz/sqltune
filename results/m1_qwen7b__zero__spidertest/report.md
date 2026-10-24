# Run `m1_qwen7b__zero__spidertest`

| metric | value |
|---|---|
| n | 2147 |
| n_gold_error | 0 |
| n_transport_error | 0 |
| ex | 74.0 |
| ex_col_tolerant | 78.7 |
| rate_correct | 74.0 |
| rate_wrong_result | 23.5 |
| rate_exec_error | 2.6 |
| rate_timeout | 0.0 |
| rate_too_many_rows | 0.0 |
| rate_no_sql | 0.0 |
| latency_p50_ms | 327.3 |
| latency_p95_ms | 794.1 |
| mean_prompt_tokens | 360.9 |
| mean_completion_tokens | 35.1 |
| generation_s | 54.5 |

## Execution errors

| kind | count |
|---|---|
| no such column | 39 |
| ambiguous column | 10 |
| syntax error | 5 |
| other | 1 |

## Per database

| db | n | EX (col-tolerant) |
|---|---|---|
| bakery_1 | 104 | 50.0 |
| cre_Doc_and_collections | 80 | 51.2 |
| aan_1 | 90 | 58.9 |
| real_estate_rentals | 72 | 63.9 |
| government_shift | 40 | 67.5 |
| planet_1 | 76 | 68.4 |
| club_leader | 17 | 70.6 |
| cre_Students_Information_Systems | 76 | 71.1 |
| boat_1 | 78 | 71.8 |
| warehouse_1 | 78 | 74.4 |
| e_commerce | 80 | 75.0 |
| movie_2 | 52 | 75.0 |
| art_1 | 114 | 75.4 |
| conference | 44 | 77.3 |
| address_1 | 80 | 78.8 |
| headphone_store | 44 | 79.5 |
| sing_contest | 20 | 80.0 |
| pilot_1 | 82 | 82.9 |
| university_rank | 44 | 84.1 |
| advertising_agencies | 90 | 84.4 |
| restaurant_bills | 30 | 86.7 |
| bbc_channels | 16 | 87.5 |
| institution_sports | 40 | 87.5 |
| book_press | 44 | 88.6 |
| online_exams | 40 | 90.0 |
| country_language | 40 | 90.0 |
| car_racing | 50 | 90.0 |
| vehicle_driver | 42 | 90.5 |
| district_spokesman | 21 | 90.5 |
| vehicle_rent | 44 | 90.9 |
| book_1 | 78 | 92.3 |
| cre_Doc_Workflow | 40 | 92.5 |
| tv_shows | 15 | 93.3 |
| bike_racing | 17 | 94.1 |
| region_building | 40 | 95.0 |
| customers_and_orders | 82 | 95.1 |
| book_review | 21 | 95.2 |
| video_game | 42 | 95.2 |
| car_road_race | 44 | 95.5 |
| soccer_3 | 40 | 97.5 |

