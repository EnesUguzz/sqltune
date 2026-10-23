# Proje 2 · M1a — BIRD, dürüst baseline'lar, eğitim taramaları · Uygulama Planı

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** M0'ın +8,9'unun ne kadarının gerçek SQL becerisi olduğunu ölçmek. Bunun için BIRD Mini-Dev'de teşhis, few-shot / örnek satır / retry baseline'ları, öğrenme eğrisi ve eğitim taramaları (epoch, train_others, LoRA vs QLoRA, seed), 7B/14B baseline'ları yapılacak. Hepsi Spider dev + `bird_diag` üzerinde ölçülür.

**Architecture:** M0 kod tabanı genişletilir. Kaynak bir adaptör (`data/bird.py`), `Example`'a `evidence`/`difficulty` alanları, prompt varyantları (`PromptConfig.mode`), runner'da tek turlu retry, birden çok LoRA adapter'ı tek vLLM sunucusunda (`--lora-modules a=… b=…`). Koşular `make` hedefleriyle sıralı, sonuçlar `results/<run>/` + `results/SUMMARY.md`.

**Tech Stack:** M0 ile aynı. vLLM v0.31.0; Qwen2.5-Coder-7B/14B-Instruct-AWQ.

**Spec:** `docs/specs/2026-10-23-proje2-text-to-sql-finetune-design.md` (§13 M1). M0 sonuç notu: `docs/decisions/001-m0-first-delta.md`.

## Global Constraints

- Commit'lerde Claude atıf satırı yok; push yok (kullanıcı onayı gerekir).
- **Claude Haiku bu planda yok.** Abonelik kotası kullanıcıyla birlikte ölçülecek (spec: "kota ölçümü → tekli/batch kararı").
- **Test bir kez kuralı:** Spider test ve `bird_holdout` bu planda koşulmaz. Kararlar Spider dev + `bird_diag` ile verilir. Final satırlar M1b'de bir kez koşulur.
- Hard-coding yok; her yeni ayar `config/models.py`'de `What/Typical/Effect` açıklamalı.
- Aynı prompt kuralı: bir satırdaki model eğitimde hangi prompt'u gördüyse eval'de de onu görür; prompt varyantı sütun olarak raporlanır.
- BIRD EX: BIRD resmi değerlendiricisi gibi **set** karşılaştırması (`EvalConfig.row_compare: set`); Spider multiset kalır.

## Review Focus

1. **BIRD veritabanları büyük ve kirli** (uzun metin, NULL, tarih formatları) → zaman aşımı/satır sınırı çalışır, gold_error sayılır, `exec_timeout_s` BIRD config'inde daha yüksek. (Test: Task 2)
2. **Few-shot örnekleri değerlendirilen DB'den sızıyor** → örnekler yalnızca Spider train'den, farklı `db_id`'den, sabit seed. (Test: Task 3)
3. **Retry, gold'u ya da sonucu görüyor** → retry mesajı yalnızca SQLite hata metnini içerir; doğru/yanlış bilgisi asla gitmez. (Test: Task 3)
4. **Birden çok adapter tek sunucuda yanlış adapter'a yönleniyor** → her adapter kendi adıyla servis edilir, `/v1/models` listesi ve koşu config'indeki `inference.model` eşleşir; rapor `config.yaml`'ı yazar. (Test: Task 5 duman kontrolü)
5. **bird_diag / bird_holdout ayrımı DB'ye göre değil** → aynı DB iki tarafta olamaz. (Test: Task 2)

---

### Task 1: M0'ı kapatma (doküman + kayıt düzeltmeleri)

**Files:** `docs/decisions/001-m0-first-delta.md`, `src/sqltune/{runner.py,evaluation/metrics.py,evaluation/report.py,evaluation/compare.py,training/train.py,data/spider.py,config/models.py}`, testler.

- 001: düzelen/bozulan sayılarının metriği (strict 151/54, tolerant 144/52); McNemar yanında **DB-küme bootstrap %95 CI [+5,7, +11,7]**; `final_train_loss` = ortalama loss (son ~0,03); "official-evaluator style" → "column-order tolerant (the official test-suite evaluator also accepts permuted columns; this is not that evaluator)"; config'teki ölçülmemiş "±1 point" iddiası kaldırılır.
- `pred_error: str | None` (SQLite hata metni) `predictions.jsonl`'e yazılır; `report.md`'ye hata taksonomisi (`no such column`, `no such table`, `ambiguous column`, `syntax`, diğer) + DB bazında EX tablosu.
- `metrics.py`: `db_cluster_bootstrap_ci(ref_rows, rows, metric, n=2000, seed) -> (lo, hi)`; rapor Δ satırına CI.
- `compare.py`: permütasyon sınırı aşılırsa sayılır (`n_perm_cap` metriği).
- `train.py`: `warmup_steps` adım sayısı olarak hesaplanır (`ceil(ratio × total_steps)`), `tokens_per_s` = (eğitilen token sayısı / süre), dataset uzunluklarından.
- `DataConfig.include_train_others: bool = False` (train'e `train_others.json` eklenir; id `spider-trainothers-NNNN`).
- `TrainConfig.train_fraction: float = 1.0` (öğrenme eğrisi; sabit seed'le alt örnekleme).
- Testler: `test_pred_error_written`, `test_error_taxonomy_in_report`, `test_cluster_ci_contains_point_estimate`, `test_include_train_others`, `test_train_fraction_deterministic`, `test_warmup_steps_from_ratio`.

### Task 2: BIRD Mini-Dev (SQLite)

**Files:** `src/sqltune/data/bird.py`, `src/sqltune/domain.py` (`evidence: str | None = None`, `difficulty: str | None = None`, `source: str = "spider"`), `data/download.py` (BIRD), `config/models.py` (`DataConfig.dataset: Literal["spider","bird"]`, `DataConfig.bird_split: Literal["diag","holdout","all"]`, `EvalConfig.row_compare: Literal["multiset","set"]`), prompt şablonu `prompts/sql_single_evidence.md` (evidence varsa "Evidence: …" satırı).

- İndirme: resmi Mini-Dev SQLite paketi (`bird-bench/mini_dev` README'deki bağlantı); gerçek yapı Step 1'de listelenip loader buna göre yazılır. JSON alanları: `question_id, db_id, question, evidence, SQL, difficulty`.
- `split_bird(examples, seed) -> (diag, holdout)`: DB'leri sabit seed'le karıştır, soruların ~yarısı diag'a gelene kadar DB ekle (DB ayrık).
- `row_compare: set` → `set(gold) == set(pred)` (BIRD resmi davranışı); tolerant varyant da set üzerinde.
- Testler: `test_bird_loader_fields` (fixture), `test_split_disjoint_dbs`, `test_split_deterministic`, `test_set_compare_ignores_duplicates`, `test_evidence_in_prompt_only_when_present`.
- Gerçek veri: 500 soru; gold'ları kendileriyle puanla (`gold_error` sayısı + süre) → README "Data".

### Task 3: Prompt varyantları ve retry

**Files:** `config/models.py` (`PromptConfig.sample_rows: int = 0`, `PromptConfig.few_shot_k: int = 0`, `PromptConfig.few_shot_seed: int = 13`, `InferenceConfig.retry_on_error: bool = False`), `prompting/schema.py`, `prompting/builder.py`, `prompting/fewshot.py`, `runner.py`, `prompts/sql_retry.md`.

- `schema_text(db_path, sample_rows)`: her tablonun altında `/* 3 example rows from <t>: ... */` (değerler 40 karakterde kesilir).
- Few-shot: Spider train'den `few_shot_k` örnek, hedef `db_id`'den **farklı** DB'lerden, sabit seed; her örnek [user: şema+soru, assistant: ```sql …```] çifti olarak system'dan sonra eklenir. (Spec: 3 örnek, aynı DB değil.)
- Retry: ilk tahmin `exec_error` ise bir tur daha: önceki mesajlar + assistant cevabı + user: `sql_retry.md` ("The query failed with this SQLite error: {error}. Return a corrected query in a ```sql block.") — yalnızca hata metni. İkinci tahmin puanlanır; `retried: bool` kaydedilir.
- Testler: `test_sample_rows_three`, `test_fewshot_excludes_target_db`, `test_fewshot_deterministic`, `test_retry_message_has_only_error_text`, `test_retry_only_after_exec_error` (sahte backend).

### Task 4: Eğitim taramaları (GPU, sıralı)

Config'ler `experiments/configs/m1/train/*.yaml` (hepsi `extends: ../../m0/qlora.yaml`):
- Öğrenme eğrisi (1 epoch): `lc_f0125`, `lc_f025`, `lc_f05`, `lc_f1` (`train_fraction` 0.125/0.25/0.5/1.0, `num_train_epochs: 1`).
- `ep3` (3 epoch), `others` (`include_train_others: true`, 2 epoch), `lora16` (`load_in_4bit: false`), `seed2`, `seed3` (2 epoch, `training.seed` 14/15).
- `make m1-train`: hepsini sırayla eğitir (her biri öncekinin bitmesini bekler; vLLM kapalı olmalı). Tahmini GPU süresi ~3,5 saat.
- Her koşu `train_summary.json`; hata olursa sıradakine geçilir ve hata loglanır.

### Task 5: Değerlendirme matrisi

- `make serve-multi ADAPTERS="m0_qlora lc_f0125 …"` → tek vLLM, her adapter kendi adıyla (`--max-loras 4`, `--max-cpu-loras` = adapter sayısı).
- Eval config'leri `experiments/configs/m1/eval/` (üretilir: `sqltune gen-eval-configs`), her (model, prompt varyantı, set) için bir dosya:
  - 3B base: zero-shot, few-shot(3), rows(3), retry, rows+retry — Spider dev + bird_diag.
  - m0_qlora: zero-shot, retry — Spider dev + bird_diag. (rows/few-shot: eğitim prompt'u değil → raporda "prompt mismatch" diye ayrı.)
  - Tarama adapter'ları: zero-shot — Spider dev + bird_diag.
- 7B / 14B AWQ: ayrı sunucu (`MODEL=Qwen/Qwen2.5-Coder-7B-Instruct-AWQ`), zero-shot + retry, Spider dev + bird_diag. 14B 16 GB'a sığmazsa `MAX_MODEL_LEN` düşürülür; olmazsa satır "sığmadı" diye raporlanır.
- `sqltune summarize` → `results/SUMMARY.md`: satır = koşu; sütunlar EX (tolerant), EX strict, exec_error, no such column payı, p50/p95, eğitim süresi/VRAM (varsa), Δ vs base zero-shot + DB-küme CI.

### Task 6: Sonuç notu

- `docs/decisions/002-m1a-bird-baselines-sweeps.md`: tablo, öğrenme eğrisi (Spider vs bird_diag), en iyi konfig önerisi (yalnızca Spider dev + bird_diag'a dayanarak), uydurma kolon hatasının değişimi, sabah kullanıcıya sorulacaklar (Haiku kotası, final konfig, M1b kapsamı).
- README'ye kısa M1a satırı. Final review (bağımsız, en güçlü model) + fix pass.
