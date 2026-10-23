# Run `m0_qlora_eval`

| metric | m0_baseline | m0_qlora_eval | Δ vs m0_baseline |
|---|---|---|---|
| n | 1034 | 1034 | +0 |
| n_gold_error | 0 | 0 | +0 |
| ex | 64.2 | 73.6 | +9.4 |
| ex_col_tolerant | 67.9 | 76.8 | +8.9 |
| rate_correct | 64.2 | 73.6 | +9.4 |
| rate_wrong_result | 25.1 | 21.5 | -3.7 |
| rate_exec_error | 10.6 | 4.9 | -5.7 |
| rate_timeout | 0.0 | 0.0 | +0.0 |
| rate_too_many_rows | 0.0 | 0.0 | +0.0 |
| rate_no_sql | 0.0 | 0.0 | +0.0 |
| latency_p50_ms | 372.9 | 423.4 | +50.5 |
| latency_p95_ms | 833.5 | 1029.3 | +195.9 |
| mean_prompt_tokens | 387.4 | 387.4 | +0.0 |
| mean_completion_tokens | 36.9 | 36.7 | -0.3 |
| generation_s | 28.6 | 34.0 | +5.5 |

Paired questions: 1034; fixed: 151; broken: 54; exact McNemar p = 8.175e-12

### Fixed (wrong before, correct now)

- **spider-dev-0006** (concert_singer, now `correct`): Show the name and the release year of the song by the youngest singer.
  - gold: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
  - pred: `SELECT Song_Name ,  Song_release_year FROM singer ORDER BY Age ASC LIMIT 1`
- **spider-dev-0007** (concert_singer, now `correct`): What are the names and release years for all the songs of the youngest singer?
  - gold: `SELECT song_name ,  song_release_year FROM singer ORDER BY age LIMIT 1`
  - pred: `SELECT Song_Name ,  Song_release_year FROM singer ORDER BY Age ASC LIMIT 1`
- **spider-dev-0008** (concert_singer, now `correct`): What are all distinct countries where singers above age 20 are from?
  - gold: `SELECT DISTINCT country FROM singer WHERE age  >  20`
  - pred: `SELECT DISTINCT country FROM singer WHERE age  >  20`
- **spider-dev-0009** (concert_singer, now `correct`): What are  the different countries with singers above age 20?
  - gold: `SELECT DISTINCT country FROM singer WHERE age  >  20`
  - pred: `SELECT DISTINCT country FROM singer WHERE age  >  20`
- **spider-dev-0012** (concert_singer, now `correct`): List all song names by singers above the average age.
  - gold: `SELECT song_name FROM singer WHERE age  >  (SELECT avg(age) FROM singer)`
  - pred: `SELECT Song_Name FROM singer WHERE Age  >  (SELECT avg(Age) FROM singer)`

### Broken (correct before, wrong now)

- **spider-dev-0002** (concert_singer, now `wrong_result`): Show name, country, age for all singers ordered by age from the oldest to the youngest.
  - gold: `SELECT name ,  country ,  age FROM singer ORDER BY age DESC`
  - pred: `SELECT name ,  country ,  age FROM singer ORDER BY age`
- **spider-dev-0011** (concert_singer, now `wrong_result`): How many singers are from each country?
  - gold: `SELECT country ,  count(*) FROM singer GROUP BY country`
  - pred: `SELECT count(*) ,  country FROM singer GROUP BY country`
- **spider-dev-0023** (concert_singer, now `wrong_result`): For each stadium, how many concerts play there?
  - gold: `SELECT T2.name ,  count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.stadium_id`
  - pred: `SELECT count(*) ,  T1.name FROM stadium AS T1 JOIN concert AS T2 ON T1.stadium_id  =  T2.stadium_id GROUP BY T1.name`
- **spider-dev-0044** (concert_singer, now `wrong_result`): What are the number of concerts that occurred in the stadium with the largest capacity ?
  - gold: `select count(*) from concert where stadium_id = (select stadium_id from stadium order by capacity desc limit 1)`
  - pred: `SELECT count(*) FROM concert AS T1 JOIN stadium AS T2 ON T1.Stadium_ID  =  T2.Stadium_ID ORDER BY T2.Capacity DESC LIMIT 1`
- **spider-dev-0067** (pets_1, now `exec_error`): Find the type and weight of the youngest pet.
  - gold: `SELECT pettype ,  weight FROM pets ORDER BY pet_age LIMIT 1`
  - pred: `SELECT pet_type ,  weight FROM Pets ORDER BY pet_age LIMIT 1`

