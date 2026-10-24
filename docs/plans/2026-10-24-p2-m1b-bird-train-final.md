# Proje 2 · M1b — Spider + BIRD train ile 3B fine-tune, final ölçüm · Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M1a'nın bulgusunu test etmek. M1a'da yalnızca Spider ile yapılan fine-tune BIRD'e taşınmadı. Bu planda Qwen2.5-Coder-3B'yi **base modelden** Spider train + BIRD train karışımıyla eğitiyoruz. Eğitim içinde veritabanı bazlı ayrılmış %5'lik bir validation izleniyor. Kararlar Spider dev + `bird_diag` ile veriliyor. Final satırlar Spider test + `bird_holdout` üzerinde bir kez koşuluyor. MLflow kayıtları ve README'de "deney takibi" bölümü de bu planın parçası.

**Architecture:** M1a kod tabanı genişletiliyor:
- İkinci kaynak adaptörü: BIRD train (`data/bird_train.py`).
- Eğitim kaynak listesi: `DataConfig.train_sources`.
- Veritabanı bazlı validation ayrımı: `subsets.split_by_db`.
- Eğitimde `eval_dataset` / `eval_loss`.
- Eval koşularında eğitim koşusuna MLflow bağlantısı.
- Test setleri için koruma: `run.test_once`.

Eval matrisi M1a'daki gibi `scripts/gen_m1_eval_configs.py` ve `make m1-eval GROUP=…` ile çalışıyor, sonuçlar aynı `results/SUMMARY.md`'ye düşüyor.

**Tech Stack:** M0/M1a ile aynı (Unsloth 2026.10.8, TRL 1.15, transformers 5.17, vLLM v0.31.0 docker, MLflow SQLite). Grafik için matplotlib (zaten `mlflow` bağımlılığıyla geliyor; yoksa `train` extra'ya eklenir).

**Spec:** `docs/specs/2026-10-23-proje2-text-to-sql-finetune-design.md`. Ek kaynaklar: M1a sonuç notu `docs/decisions/002-m1a-bird-baselines-sweeps.md`; kullanıcı kararları (2026-10-24, aşağıda).

## Global Constraints

- **Spec değişikliği (kullanıcı kararı, 2026-10-24):** spec §14 "BIRD train ile eğitim"i kapsam dışı sayıyordu. Kullanıcı M1b'de bunu kapsama aldı. Tek ana koşu: Spider train + BIRD train, base modelden (M0 adapter'ının üstüne değil). 7B fine-tune ve Claude Haiku bu planda yok; ikisine M1b sonuçlarına bakılarak karar verilecek.
- Commit'lerde Claude atıf satırı yok. Push yalnızca kullanıcı onayıyla (repo: `github.com/EnesUguzz/sqltune`).
- **Test bir kez kuralı:** Spider test ve `bird_holdout`, yalnızca Task 6'da ve final satırlar için birer kez koşulur. Bunu kod korur (`run.test_once`).
- **Veri ayrımı:**
  - BIRD dev'in Mini-Dev dışındaki soruları eğitimde kullanılmaz (aynı 11 veritabanı).
  - Validation, train veritabanlarından **bütün veritabanı** olarak ayrılır.
  - Hiçbir eğitim veya validation veritabanı Spider dev/test ya da Mini-Dev'de bulunamaz.
- Hard-coding yok. Her yeni ayar `config/models.py`'de `What/Typical/Effect` açıklamalı olmalı.
- Aynı prompt kuralı: Spider eğitim kayıtlarının prompt'u M0 ile byte byte aynı kalır. BIRD kayıtları Mini-Dev eval'iyle aynı `evidence_template`'i kullanır. Eğitimde `sample_rows=0`.
- BIRD EX: `row_compare: set`, `exec_timeout_s: 30`, `gold_timeout_s: 120`. Tablolarda strict EX de gösterilir (yayınlanmış BIRD sayılarıyla kıyaslanabilir olan budur).
- M1a'daki ayar taramaları seed gürültüsü içinde kaldı. M1b'de hiperparametre taraması yok; M0 ayarları korunur, `num_train_epochs: 1` (M1a `lc_f1`: 1 epoch ≈ 2 epoch).

## Review Focus

1. **BIRD train veritabanı Mini-Dev'de, Spider dev'de veya Spider test'te de var.** Eğitim test veritabanını görmüş olur. Kesişimin boş olduğunu kod doğrular. (Test: Task 1 `test_bird_train_dbs_disjoint_from_eval`)
2. **BIRD train gold SQL'leri gürültülü veya çok yavaş** (bilinen sorun). Çalışmayan gold'lar bir kez tespit edilir, eğitimden çıkarılır, sayısı raporlanır. (Test: Task 1 `test_failing_gold_ids_are_skipped`)
3. **Uzun BIRD şemaları `max_seq_length`'i aşıyor ve sessizce atılıyor.** Atılan kayıt sayısı kaynak başına sayılır ve raporlanır. `max_seq_length` ölçüme göre seçilir. (Test: Task 2 `test_dropped_counted_per_source`)
4. **Validation soru bazında ayrılıyor veya bir kaynak hiç validation almıyor.** Ayrım veritabanı bazında, deterministik ve kaynak başına yapılır. (Test: Task 2 `test_split_by_db_*`)
5. **Geliştirme sırasında test seti yanlışlıkla koşuluyor.** `run.test_once: true` olmayan bir config Spider test veya `bird_holdout` yüklerse runner hata verir. (Test: Task 6 `test_test_sets_need_test_once`)

## Revizyon 1 — uzman incelemesi (2026-10-24, kullanıcı onaylı)

Aşağıdaki maddeler, task metinleriyle çelişirse **task metninin önüne geçer**.

1. **Eğitimde karışık örnek satır.**
   - Yeni alan: `TrainConfig.train_sample_rows_mix: float = 0.0` (m1b'de 0.5). Her eğitim kaydı seed'le belirlenen bu olasılıkla `sample_rows=3` prompt'unu, aksi halde `sample_rows=0` prompt'unu alır. Validation kayıtları da aynı kuralla üretilir.
   - Böylece `zero` ve `rows3retry` değerlendirmeleri ikisi de eğitim dağılımında olur.
   - Test: `test_rows_mix_assigns_both_prompts` (Task 2).
2. **Uzunluk kuralı.**
   - `max_seq_length`, BIRD'ü ~%100 tutan en küçük değer olur: 4096 ile 6144 arası, karışık örnek satırlı kayıtlarla ölçülür.
   - VRAM için `per_device_train_batch_size: 2`, `gradient_accumulation_steps: 8` (efektif batch 16).
   - Atılan kayıtlar **veritabanı bazında** `train_summary.json`'a yazılır (`dropped_by_db`).
   - Şema budama yok (eval tam şema kullanıyor).
3. **Sızıntı kontrolü şema parmak iziyle.** Task 1'deki ayrıklık testi `db_id`'nin yanında küçük harfli tablo adı kümelerini de karşılaştırır. Bir eğitim veritabanının tablo kümesinin, herhangi bir dev/test veritabanınınkiyle Jaccard benzerliği ≥ 0,5 ise rapor edilir. Eşik aşılırsa o veritabanı eğitimden çıkarılır ve Ruling yazılır.
4. **İstatistik.**
   - 003 notunda her Δ için soru bazında eşleştirilmiş McNemar p değeri verilir.
   - 5 veritabanlı CI'lar "gösterge" olarak etiketlenir.
   - **Önceden yazılan karar kuralı:** `bird_diag`'de < 5 puanlık fark kazanç sayılmaz.
5. **Hata analizi.** Task 5'e eklenir: m1b_mix'in `bird_diag` hatalarından seed'le 40 soru seçilir ve elle kategorilere ayrılır (şema eşleme / değer-format / join / aggregation / evidence'ı yok sayma / gold şüpheli). Ayrıca Mini-Dev zorluk seviyesine göre EX hesaplanır.
6. **README uyarıları** (Task 7):
   - `bird_holdout` = Mini-Dev'in 6 veritabanlık alt kümesi; yayınlanmış Mini-Dev/BIRD-dev skorlarıyla doğrudan kıyaslanamaz.
   - Spider test EX bizim harness'ımızla ölçüldü (test-suite değil).
   - Qwen2.5-Coder ön eğitiminde olası kirlenme (mutlak skorlar şişik olabilir; aynı base'e göre Δ geçerli).
   - BIRD gold'larında anlamsal gürültü var.
7. **Çıkarılanlar:** M1a koşularına MLflow tag'i ekleyen geriye dönük script (Task 3 Step 5, ikinci madde) yapılmaz. MLflow ekran görüntüsü **kalır** (kullanıcı MLflow'un GitHub'da gösterilmesini istedi).
8. **İsteğe bağlı** (zaman kalırsa):
   - Task 5'te 50 prompt'luk vLLM ve HF+PEFT çıktı karşılaştırması.
   - Task 6'da `results/<run>/.test_once_done` işaret dosyası ile ikinci koşunun engellenmesi.
   - Request uzunluğu 8192'yi aşan istek sayısının metriklerde 0 olduğunun doğrulanması.
9. **Gece öncesi hazırlık (kullanıcı isteği):** Task 1–3 tamamlanır, ardından duman eğitimi, Spider ve BIRD veri seti incelemesi (`docs/data/dataset-review.md`) ve bağımsız kod/veri incelemesi yapılır. Geriye yalnızca Task 4 Step 1'in komutu kalır.

---

### Task 1: BIRD train indirme, yükleme, gold kontrolü

**Files:**
- Create: `src/sqltune/data/bird_train.py`, `tests/test_bird_train.py`, `scripts/check_bird_train_gold.py`
- Modify: `src/sqltune/data/download.py` (`download_bird_train`), `src/sqltune/cli.py` (`download` komutu BIRD train'i de indirir ve sayıları basar), `README.md` (Data bölümü), `Makefile` (`bird-train-gold`)

**Interfaces:**
- Produces:
  - `download_bird_train(cfg: DataConfig) -> Path` → `<data_dir>/bird_train/{train.json, train_databases/<db>/<db>.sqlite}`
  - `load_bird_train(cfg: DataConfig) -> list[Example]`. Id biçimi `birdtrain-00000`, `split="train"`; boş evidence `None` olur; `difficulty=None`. `<data_dir>/bird_train/gold_failures.json` içindeki id'ler atlanır.
  - `GOLD_FAILURES_FILE = "gold_failures.json"`

- [ ] **Step 1: Failing testleri yaz** (`tests/test_bird_train.py`; küçük bir fixture, `test_bird.py`'deki `_bird_dir` kalıbıyla):
  - `test_bird_train_loader_fields`: id `birdtrain-00000`, `split == "train"`, `db_path.exists()`, boş evidence → `None`.
  - `test_failing_gold_ids_are_skipped`: `gold_failures.json = ["birdtrain-00001"]` → yüklenen id'ler arasında yok, toplam 1 eksik.
  - `@pytest.mark.data test_bird_train_dbs_disjoint_from_eval`: gerçek veride BIRD train `db_id`'leri ∩ (Mini-Dev ∪ Spider dev ∪ Spider test `db_id`'leri) = ∅; soru sayısı 9.428 (gold_failures uygulanmadan).
- [ ] **Step 2: Çalıştır, FAIL gör.** `uv run pytest -q tests/test_bird_train.py` → `ModuleNotFoundError: sqltune.data.bird_train`.
- [ ] **Step 3: `download_bird_train` yaz.**
  - Kaynak: `BIRD_TRAIN_ZIP_URL = "https://bird-bench.oss-cn-beijing.aliyuncs.com/train.zip"` → `data/raw/bird_train.zip`.
  - Zip'in içinde `train/train.json` ve iç içe `train/train_databases.zip` beklenir. Yalnızca `.sqlite` dosyaları `train_databases/<db>/<db>.sqlite` olarak çıkarılır. Üye yolunda `..` varsa reddedilir.
  - Gerçek zip yapısı farklıysa **ledger'a Ruling yaz** ve yolları ona göre düzelt.
  - İndirme boyutunu ve süresini ledger'a yaz.
- [ ] **Step 4: `load_bird_train` yaz.** Alanlar Mini-Dev ile aynı: `question`, `SQL`, `evidence`, `db_id`.
- [ ] **Step 5: Testleri çalıştır, PASS.** Komut: `uv run pytest -q tests/test_bird_train.py -m "data or not data"`; önce `uv run sqltune download` çalışmış olmalı.
- [ ] **Step 6: Gold kontrolü.** `scripts/check_bird_train_gold.py` bütün gold'ları süreç havuzunda çalıştırır.
  - `executor.execute`'u kullanır, zaman aşımı `gold_timeout_s` (120 s).
  - Hata veren, zaman aşımına düşen veya `too_many_rows` olanların id'lerini `gold_failures.json`'a yazar.
  - Hata türüne göre sayıları basar.
  - `make bird-train-gold` ile çalışır.
  - Beklenen: hataların toplamı %5'in altında. Üstündeyse ledger'a not düş ve ilk 10 örneğe bak.
- [ ] **Step 7: README Data bölümü.** BIRD train: soru ve veritabanı sayısı, indirme boyutu, düşen gold sayısı, ayrıklık testi.
- [ ] **Step 8: Commit.** `feat(m1b): BIRD train download, loader and gold check`

### Task 2: Kaynak karışımı, veritabanı bazlı validation, eval_loss

**Files:**
- Modify: `src/sqltune/config/models.py`, `src/sqltune/data/subsets.py`, `src/sqltune/training/sft_data.py`, `src/sqltune/training/train.py`
- Create: `scripts/measure_train_lengths.py`
- Test: `tests/test_sft_data.py`, `tests/test_subsets.py` (yeni)

**Interfaces:**
- Consumes: `load_bird_train` (Task 1), `load_spider(cfg, "train")`, `select_fraction`.
- Produces (config):
  - `DataConfig.train_sources: list[Literal["spider", "bird"]] = ["spider"]`
  - `TrainConfig.val_fraction: float = 0.0` (0 = validation yok; M0/M1a davranışı)
  - `TrainConfig.eval_steps: int = 200`
- Produces (kod):
  - `split_by_db(examples, fraction: float, seed: int) -> tuple[list[Example], list[Example]]`: (train, val). Veritabanları `random.Random(seed)` ile karıştırılır. Val soru sayısı `fraction × len`'e ulaşana kadar sırayla bütün veritabanları val'a alınır. Bir veritabanı iki tarafta bulunamaz.
  - `build_sft_dataset(cfg, tokenizer) -> SftData`. Alanlar:
    - `train: Dataset`
    - `val: Dataset | None`
    - `dropped: dict[str, int]` (kaynak → uzun diye atılan kayıt sayısı)
    - `counts: dict[str, dict[str, int]]` (kaynak → {"train": n, "val": n})
  - `train_summary.json` ve MLflow'a eklenen anahtarlar: `n_dropped_overlong_<source>`, `n_train_<source>`, `n_val_<source>`, `final_eval_loss`, `mlflow_run_id`.

- [ ] **Step 1: Failing testleri yaz.**
  - `test_split_by_db_disjoint_and_deterministic`: iki tarafın `db_id` kümeleri ayrık; aynı seed'le aynı sonuç; val payı ≥ fraction, ve fraction + en büyük tek veritabanının payı'ndan küçük.
  - `test_split_by_db_zero_fraction`: `fraction=0` → val boş.
  - `test_split_per_source`: karışık girdide (spider + bird fixture) iki kaynağın da val'da en az bir veritabanı var. Ayrım kaynak başına uygulanır.
  - `test_dropped_counted_per_source`: çok kısa `max_seq_length` → `dropped["bird"] > 0` ve `"spider"` anahtarı var.
  - `test_spider_records_unchanged`: `train_sources=["spider"]`, `val_fraction=0` → kayıtlar M0'daki `to_sft_record` çıktısıyla aynı (ilk 5 kaydın prompt+completion eşitliği).
- [ ] **Step 2: Çalıştır, FAIL gör** (`ImportError: split_by_db`, `AttributeError: train_sources`).
- [ ] **Step 3: Yaz.**
  - Config alanları `What/Typical/Effect` açıklamalı olacak.
  - `split_by_db` `subsets.py`'ye gider.
  - `build_sft_dataset` sırası: kaynak başına önce `select_fraction`, sonra `split_by_db(val_fraction, training.seed)`, sonra uzunluk filtresi. Train kayıtları kaynaklar karışık olacak şekilde seed'le karıştırılır.
  - `train.py`:
    - `eval_dataset=val`, `eval_strategy="steps"` (val yoksa `"no"`), `eval_steps`, `per_device_eval_batch_size = per_device_train_batch_size`.
    - Summary'ye yeni anahtarları ve `mlflow.active_run().info.run_id`'yi yaz.
    - `n_dropped_overlong` toplamı geriye dönük uyumluluk için kalır.
- [ ] **Step 4: PASS + tüm suite.** `uv run pytest -q`; ardından `uv run ruff check src tests`, `uv run mypy src`.
- [ ] **Step 5: Uzunluk ölçümü.** `scripts/measure_train_lengths.py` iki kaynağın kayıtlarını gerçek tokenizer'la ölçer. Kaynak başına p50, p95 ve max'ı, ayrıca 2048 / 3072 / 4096'da atılacak kayıt sayısını basar.
  - **Karar kuralı:** BIRD kayıtlarının ≥%97'sini tutan en küçük değer seçilir.
  - Seçilen değer ledger'a `Ruling` olarak yazılır.
- [ ] **Step 6: Duman eğitimi.** `max_steps: 20`, `eval_steps: 10` → eval_loss MLflow'da görünür; loss-mask kontrolü geçer; tepe VRAM ledger'a yazılır.
  - Seçilen uzunlukta VRAM 14 GiB'ı aşarsa `per_device_train_batch_size` yarıya iner, `gradient_accumulation_steps` iki katına çıkar (efektif batch 16 kalır). Bu da Ruling olarak yazılır.
- [ ] **Step 7: Commit.** `feat(m1b): train source mix, database-disjoint validation split, eval loss`

### Task 3: MLflow bağlantısı ve temizlik

**Files:**
- Modify: `src/sqltune/evaluation/tracking.py`
- Test: `tests/test_tracking.py` (yeni)

**Interfaces:**
- Consumes: `results/<adapter>/train_summary.json["mlflow_run_id"]` (Task 2).
- Produces: `train_tags(cfg: AppConfig) -> dict[str, str]`. Değerlendirilen model bir adapter ise, yani `results/<inference.model>/train_summary.json` varsa, `{"train_run": <model>, "train_mlflow_run_id": <id veya "">}` döner; yoksa `{}`. `log_to_mlflow` bu tag'leri ve `{"eval_set": ..., "prompt_variant": ...}` tag'lerini yazar. Varyant run adından ayrıştırılır (`m1_<model>__<variant>__<set>`); ad bu kalıpta değilse varyant tag'i yazılmaz.

- [ ] **Step 1: Failing test.**
  - `test_train_tags_for_adapter`: geçici results klasöründe `train_summary.json` (`mlflow_run_id: "abc"`) varsa → `{"train_run": "lc_f1", "train_mlflow_run_id": "abc"}`.
  - `test_train_tags_for_base`: summary yoksa → `{}`.
- [ ] **Step 2: FAIL gör. Step 3: yaz. Step 4: PASS + suite.**
- [ ] **Step 5: Temizlik (kod değil, tek seferlik).**
  - Elektrik kesintisinde yarım kalan `lora16` koşusu (m1a deneyinde, status `RUNNING`, metrik yok) `MlflowClient().set_terminated(run_id, "FAILED")` ile kapatılır ve ona `note: interrupted by power cut; retrained` tag'i eklenir. Yapılanı ledger'a yaz.
  - M1a eval koşularına geriye dönük `train_run` tag'i: aynı `train_tags` mantığıyla tek seferlik bir script (`outputs/` altında, commit edilmez).
- [ ] **Step 6: Commit.** `feat(m1b): link eval runs to their training run in MLflow`

### Task 4: m1b_mix eğitimi

**Files:**
- Create: `experiments/configs/m1b/train/mix.yaml`

```yaml
# M1b main run: Spider train + BIRD train, base model, 1 epoch, 5% DB-disjoint validation.
extends: ../../m0/qlora.yaml
run: {experiment_name: m1b, run_name: m1b_mix}
data: {train_sources: [spider, bird]}
training:
  num_train_epochs: 1
  val_fraction: 0.05
  max_seq_length: <Task 2 Step 5 kararı>
```

- [ ] **Step 1:** vLLM kapalı (`make serve-down`). `make train CONFIG=experiments/configs/m1b/train/mix.yaml > outputs/m1b_mix.log 2>&1` arka planda çalışır.
  - Beklenen süre 45–90 dk.
  - Kesinti olursa en son checkpoint'ten devam edilmez; koşu sıfırdan başlatılır (M1a'daki gibi) ve Ruling olarak yazılır.
- [ ] **Step 2: Kontrol.**
  - `results/m1b_mix/train_summary.json`: `n_train_spider`, `n_train_bird`, `n_val_*`, `n_dropped_overlong_*`, `final_eval_loss`.
  - MLflow'da train/eval loss eğrisi. Beklenen: eval_loss sonda yükselmiyor. Yükseliyorsa not düşülür, koşu tekrarlanmaz.
- [ ] **Step 3: Commit.** Config ve `results/m1b_mix/train_summary.json`: `experiments(m1b): train m1b_mix`

### Task 5: Dev ölçümleri ve karar

**Files:**
- Modify: `scripts/gen_m1_eval_configs.py`. Yeni grup `m1b`: `m1b_mix` × {zero, rows3retry} × {spider, birddiag}. `base` × {zero, rows3retry} × birddiag yeniden üretilir; M1a'daki `bird-0393` gold tutarsızlığı giderilmiş olur. Makefile `ADAPTERS` listesine `m1b_mix` eklenir.
- Create: `docs/decisions/003-m1b-bird-train.md`

- [ ] **Step 1:** `make serve-multi ADAPTERS="m0_qlora m1b_mix"` → `/v1/models` listesinde `base`, `m0_qlora`, `m1b_mix` görünür.
- [ ] **Step 2:** `make m1-eval GROUP=m1b` → 6 koşu. Hepsinde `n_transport_error == 0` olmalı. Ardından `make serve-down` ve `make summarize`.
- [ ] **Step 3: 003 notu (dev).** İçerik:
  - m1b_mix'in base zero'ya ve m0_qlora'ya göre Δ'sı, DB-küme CI'ı ile (bird_diag için "gösterge niteliğinde" notuyla).
  - Spider dev'de düşüş var mı.
  - `no such column` oranı.
  - Validation loss eğrisinin yorumu.
  - **Final satır listesi (Task 6'ya girdi):** base 3B, m0_qlora, m1b_mix, 7B AWQ, 14B AWQ; her biri `zero` ve `rows3retry` ile.
- [ ] **Step 4: Commit.** `docs(m1b): dev results and final row selection`

### Task 6: Test bir kez — Spider test + bird_holdout

**Files:**
- Modify:
  - `src/sqltune/data/spider.py`: `load_spider(cfg, split)` `"test"` kabul eder (`test.json`, `test_database/`).
  - `src/sqltune/config/models.py`: `RunConfig.test_once: bool = False` ve `DataConfig.split` Literal'ine `"test"` eklenir.
  - `src/sqltune/runner.py`: `load_examples` başında koruma.
  - `scripts/gen_m1_eval_configs.py`: `SETS`'e `spidertest`, `birdholdout`; `final` grupları yalnızca bu setlerle ve `run.test_once: true` ile üretilir.
- Test: `tests/test_runner.py`, `tests/test_spider.py`

**Interfaces:**
- Produces: `TestSetGuardError(RuntimeError)`. `load_examples(cfg)` `dataset=spider, split=test` veya `dataset=bird, bird_split=holdout` için `run.test_once` False ise bu hatayı fırlatır.

- [ ] **Step 1: Failing testler.**
  - `test_test_sets_need_test_once`: holdout config + `test_once=False` → `TestSetGuardError`; `test_once=True` → yüklenir.
  - `test_load_spider_test_split`: fixture'da `test.json` + `test_database/` → `split == "test"`, `db_path` `test_database` altında.
- [ ] **Step 2: FAIL. Step 3: yaz. Step 4: PASS + suite.**
- [ ] **Step 5: Koşular.** Final satırlar × {spidertest, birdholdout}:
  - `make serve-multi ADAPTERS="m0_qlora m1b_mix"` → `GROUP=final3b` (3 model × 2 varyant × 2 set = 12 koşu).
  - 7B → `GROUP=final7b` (4 koşu).
  - 14B, `GPU_UTIL=0.88` ve `--enforce-eager` ile → `GROUP=final14b` (4 koşu).
  - Spider test 2.147 soru; tahmini toplam ~1 saat.
  - Her koşu yalnızca bir kez. Bir koşu bağlantı hatasıyla bozulursa (`n_transport_error > 0`) aynı config tekrar koşulur. Bu "ikinci bakış" sayılmaz, ledger'a yazılır.
- [ ] **Step 6:** `make summarize`, ardından commit: `experiments(m1b): test-once runs on Spider test and bird_holdout`.

### Task 7: Rapor, deney takibi bölümü, son inceleme

**Files:**
- Create: `scripts/plot_training_curves.py`, `docs/img/m1b_loss.png`, `docs/img/m1_learning_curve.png`, `docs/img/mlflow_compare.png`
- Modify: `README.md`, `docs/decisions/003-m1b-bird-train.md`

- [ ] **Step 1: `plot_training_curves.py`.** MLflow'dan (`MlflowClient.get_metric_history`) `loss` ve `eval_loss` geçmişini okur.
  - Çıktı 1: `m1b_loss.png` (m1b_mix'in train/eval loss'u, karşılaştırma için m0_qlora'nın train loss'u).
  - Çıktı 2: `m1_learning_curve.png` (M1a öğrenme eğrisi: veri oranı → Spider dev / bird_diag EX, `results/m1_*`'den).
  - Kullanılan run adları script başında sabit liste olarak durur (docstring'de açıklanır).
- [ ] **Step 2: MLflow ekran görüntüsü.** `uv run mlflow ui` → playwright ile "Compare runs" görünümü (m0_qlora, lc_f1, m1b_mix), `docs/img/mlflow_compare.png`. Ekran görüntüsü alınamazsa Ruling yazılır ve yalnızca grafiklerle devam edilir.
- [ ] **Step 3: README.**
  - Ana tablo **test sonuçları** (spec §2: "README'nin ana tablosu test sonuçlarıdır"). Satırlar final modeller × varyant; sütunlar Spider test EX, `bird_holdout` EX (strict ve tolerant), eğitim süresi, VRAM.
  - Dev tablosu ayrı, `results/SUMMARY.md` linkiyle.
  - Yeni bölüm **"Experiment tracking"**: hangi parametre ve metriklerin kaydedildiği (eğitim: tüm config, loss/eval_loss eğrisi, süre, VRAM, kaynak başına kayıt ve atılan sayıları; eval: tüm config, EX, hata türleri, gecikme, tahmin dosyaları ve eğitim koşusuna bağlantı), iki grafik ve ekran görüntüsü, `mlflow ui` komutu.
  - `mlflow.db` repoda yok; bu da yazılır.
- [ ] **Step 4: 003 notu (final).** Test tablosu, dev ile test arasındaki fark ve 7B fine-tune / Haiku kararı için soru listesi.
- [ ] **Step 5: Commit.** `docs(m1b): final results, experiment tracking section, figures`
- [ ] **Step 6: Bağımsız son inceleme.** opus modeliyle, M1a'daki gibi review package üzerinden. Önemli bulgular tek düzeltme turunda TDD ile kapatılır.
- [ ] **Step 7: Push için kullanıcı onayı istenir.**
