# Proje 2 — Küçük Modelle Text-to-SQL Fine-Tuning (Spider) · Tasarım Spec'i

- **Tarih:** 2026-10-23
- **Durum:** Taslak — kullanıcı onayı bekliyor
- **Çalışma adı:** `sqltune` (Claude'un önerisi; repo adı sonra değiştirilebilir)
- **Repo:** yeni, ayrı public repo (`github.com/EnesUguzz/<ad>`), paperlens'ten bağımsız

---

## 1. Amaç ve neden

**Amaç:** Fine-tuning'i baştan sona hızlı ve anlayarak öğrenmek: veri hazırlama → PEFT (LoRA/QLoRA) ile eğitim → nesnel değerlendirme → modeli servis etmek (deploy) → "fine-tune emeğe değdi mi" sorusunu sayılarla cevaplamak.

**Neden text-to-SQL:** doğruluk nesnel ve ucuz ölçülür (SQL çalıştırılır, sonuç kümeleri karşılaştırılır; LLM-judge yok), Spider hazır ve temiz, görev iş ilanlarında karşılığı olan bir problem, ve aynı doğruluk kontrolü ileride GRPO için hazır ödül fonksiyonu olur.

**Ürün çerçevesi:** "Şemanın ve verinin dışarı çıkamadığı bir şirkette self-servis SQL asistanı."

**Hikaye (README'nin ilk paragrafı — sonuç + trade-off):** "Tek bir tüketici GPU'sunda ~1 saatlik QLoRA ile 3B model, BIRD Mini-Dev'de 14B'nin / Claude Haiku'nun doğruluğunun %X'ine ulaşıyor; gecikmesi Y kat düşük, 1.000 sorgu başına maliyeti ~0, veri makineden çıkmıyor." Başlık sayıları **BIRD Mini-Dev**'den gelir (Spider eski ve muhtemelen sızmış; Spider ayar seti olarak kalır).

*(2026-10-23 bağımsız PM incelemesi sonrası güncellendi: dikey dilim sıralaması, few-shot/retry satırları, maliyet + Pareto grafiği, anlamsal hata analizi, demo.)*

**Öncelik:** hızlı deneme döngüsü. Bir eğitim koşusu hedef olarak ≤ ~1 saat, böylece günde birkaç deneme yapılabilir.

## 2. Başarı kriteri (proje bitti sayılır, eğer)

1. README'de karşılaştırma tablosu (Spider dev, final konfig için Spider test bir kez; BIRD Mini-Dev genelleme):

   | Model | Yöntem | EX | Test-suite (Spider) | p50/p95 gecikme | VRAM |
   |---|---|---|---|---|---|
   | Qwen2.5-Coder-3B-Instruct | prompt (baseline) | | | | |
   | Qwen2.5-Coder-3B-Instruct | few-shot (3 örnek, train'den aynı DB değil) | | | | |
   | Qwen2.5-Coder-3B-Instruct | prompt + tek seferlik hata-geri-bildirimli yeniden deneme | | | | |
   | Qwen2.5-Coder-3B-Instruct | QLoRA SFT | | | | |
   | Qwen2.5-Coder-3B-Instruct | LoRA SFT | | | | |
   | Qwen2.5-Coder-7B-Instruct | prompt | | | | |
   | Qwen2.5-Coder-14B-Instruct (4-bit) | prompt | | | | |
   | *(zaman kalırsa)* Qwen3.5-4B / 9B, thinking kapalı | prompt | | | | |
   | Claude Haiku (`claude -p`) | prompt, tek seferlik | | | yok (bulut) | yok |
   | *(zaman kalırsa)* FINER-SQL-3B-Spider (hazır checkpoint) | referans tavan: SFT + RL + 37 bin sentetik veri | | | | |

   **Ek sütun:** 1.000 sorgu başına maliyet (yerel: ölçülen GPU-saat × kartın gücü × elektrik fiyatı, config'den; Haiku: liste fiyatı × ölçülen token). **Grafik:** doğruluk–gecikme Pareto grafiği (nokta boyutu = VRAM).

   **Claude satırı — tekli mi batch mi (kullanıcı kararı, 2026-10-23):** M1 başında kota küçük bir denemeyle ölçülür (~20 tekli çağrı); ~400 tekli çağrı kotaya sığıyorsa Haiku diğer modeller gibi **tek tek** sorulur (batch ve kontrol deneyi gereksizleşir). Sığmıyorsa aşağıdaki batch kurgusu kullanılır: Spider dev'den 200 + BIRD Mini-Dev'den 200 soru, **veritabanına göre gruplanmış 20'şerlik batch'ler** (toplam ~20 çağrı; batch boyutu config'den). Her soruya id verilir, cevap JSON listesi (`[{"id": ..., "sql": ...}]`) olarak istenir; koşucu id'leri doğrular, eksik/bozuk olanları tek tek yeniden sorar. **Kontrol deneyi:** 40 soru hem tek tek hem batch halinde sorulur, fark raporlanır (batch'in doğruluğa etkisi). Tabloda Claude ile diğer modeller ayrıca aynı 200 + 200 soruluk alt küme üzerinde de karşılaştırılır. Koşucu cache'li, kaldığı yerden devam eder. Spider 2018'den beri açık olduğu için Claude'un eğitim verisinde görmüş olma ihtimali README'de not edilir (BIRD Mini-Dev ve Chinook bu yüzden Claude için daha adil karşılaştırma).

2. Fine-tune edilmiş model vLLM ile OpenAI uyumlu API olarak servis ediliyor (`make serve` + Docker Compose). GGUF/Ollama isteğe bağlı.
3. Hata türü kırılımı + **anlamsal hata analizi**: 30–50 hata (yanlış JOIN, yanlış tablo/kolon seçimi, eksik/yanlış filtre, yanlış toplama/GROUP BY, değer hatası…) kategorilere ayrılır, base ve SFT yan yana; etiketleri Claude önerir, kullanıcı kontrol eder. Mekanik kırılım: çalışmayan SQL (sözdizimi / olmayan kolon), çalışan ama yanlış sonuç, zaman aşımı — baseline ve fine-tune için yan yana.
4. Gradio demosu (Chinook + Spider DB'leri) ve README'de ~30 sn GIF.
5. Öğrenme notebook'u: düz `transformers + peft + TRL` ile aynı eğitimin adım adım hali + Unsloth ile hız/VRAM karşılaştırması.
6. Kullanıcı her modülün ve her parametrenin ne yaptığını kod içi dokümantasyondan okuyup anlatabiliyor.

**Tablonun kuralları (Claude'un kararları):**
- **Test bir kez = model başına bir kez.** Tablodaki her satır, tüm ayarlar dev'de dondurulduktan sonra Spider test ve BIRD Mini-Dev'de birer kez koşulur. README'nin ana tablosu test sonuçlarıdır; dev sonuçları ayrı tabloda.
- **Test-suite sütunu:** resmi test-suite varyant veritabanları yalnızca dev için yayınlandıysa (kurulumda doğrulanır), test-suite sütunu dev tablosunda yer alır, test tablosunda yalnızca EX bulunur.
- **VRAM iki ayrı sütun:** (a) eğitim tepe VRAM'i (`torch.cuda.max_memory_allocated`, yalnızca fine-tune satırları), (b) çıkarım için model ağırlık belleği (vLLM'in yükleme logundan; KV cache hariç). vLLM'in ayırdığı toplam havuz anlamsız olduğu için raporlanmaz.
- **Gecikme:** tek istek p50/p95 (eşzamanlılık 1) + sabit eşzamanlılıkta (config, varsayılan 16) throughput (soru/saniye). Tüm satırlar mümkünse aynı vLLM sunucusunda; bir satır Ollama'da ölçülmek zorunda kalırsa tabloda işaretlenir.
- **Fine-tune satırları birleştirilmiş (merged) bf16 model** olarak servis edilir (QLoRA da: adapter bf16 tabana birleştirilir). `--enable-lora` yolu geliştirme sırasında hızlı karşılaştırma için kullanılır.
- **Düşünme (thinking/reasoning) modu kapalı:** tüm modeller doğrudan SQL üretir. SFT hedefi çıplak SQL olduğu için karşılaştırma ancak böyle adil; seçilen aile thinking modu destekliyorsa kapatılır.
- **Prompt baseline'ları zero-shot, SFT ile birebir aynı prompt'la.** Few-shot / prompt mühendisliği yapılmaz; bu, README'de sınırlama olarak belirtilir.

Mutlak bir skor hedefi yok; tahmini beklenti Spider dev'de fine-tune ile +5–20 puan, BIRD'de daha az. Kazanç küçük ya da negatif çıkarsa bu dürüstçe bulgu olarak raporlanır.

## 3. Kısıtlar

| Kısıt | Sonuç |
|---|---|
| Ücretli API ve cloud yok | Her şey yerel. W&B yerine MLflow (kullanıcı zaten biliyor). **İstisna:** Claude Haiku karşılaştırması kullanıcının Claude aboneliği üzerinden headless Claude Code CLI (`claude -p`) ile yapılır; API anahtarı yok. |
| RTX 5080 16 GB VRAM, 48 GB RAM, Ryzen 9 9950X3D | Eğitilen model 1,5–4B (LoRA 16-bit de sığar). 7B/14B yalnızca çıkarım (inference) için, quantize. |
| Blackwell (sm_120) | CUDA 12.8+; PyTorch, Unsloth, vLLM, bitsandbytes sürümleri kurulumun ilk görevinde doğrulanır. |
| Windows + WSL2 Ubuntu, Docker WSL2'de | Geliştirme WSL2 içinde (paperlens ile aynı). |
| CI'da GPU yok | CI'da yalnızca GPU gerektirmeyen testler (prompt kurucu, SQL değerlendirici, config). |
| Kullanıcı kodu elle yazmayacak | Claude yazar, kullanıcı ayarlar/inceler (§4). |
| Hız > titizlik | Tek temiz kurulum + dürüst karşılaştırma; ek titizlik isteğe bağlı. |

## 4. Çalışma modeli (paperlens ile aynı kurallar)

- **Hard-coding yok:** tüm ayarlanabilir değerler (model adı, `r`, `lora_alpha`, `learning_rate`, epoch, `max_seq_length`, prompt şablonu seçimi, zaman aşımı…) pydantic-settings modellerinde; değerler YAML deney dosyalarından gelir.
- **Her parametre kendini anlatır:** `Field(description=...)` içinde (1) ne yapar, (2) tipik aralık, (3) artarsa/azalırsa ne değişir (doğruluk, süre, VRAM). `make explain-config` hepsini tablo olarak yazdırır. Deney YAML'larında satır yanı yorumlar.
- **Her dosya kendini anlatır:** modül docstring'i (ne yapar, akıştaki yeri, okuduğu parametreler, denemeye değer 1–3 fikir); her alt pakette kısa `README.md`.
- **Deney raporu neyin değiştiğini söyler:** her koşu MLflow'a ve `results/<run_id>.md`'ye yazılır: referansa göre değişen parametreler, metrik farkları (Δ), zorluk seviyesine göre kırılım (Spider'ın easy/medium/hard/extra etiketleri), hata türleri, en çok kötüleşen/iyileşen 5 örnek (soru, gold SQL, üretilen SQL, sonuç).
- **Dil:** kod, docstring, config açıklamaları, README İngilizce; spec ve planlar Türkçe.
- **Git:** commit'lerde ve PR'larda Claude atıf satırı yok. Push her seferinde kullanıcı onayıyla.

## 5. Veri

| Kullanım | Veri | Not |
|---|---|---|
| Eğitim | Spider train (7.000 soru, 140 DB; `train_others` 1.659 örnek config ile isteğe bağlı) | Soru+SQL: HF `xlangai/spider`; SQLite veritabanları: resmi Spider zip'i (Google Drive, yale-lily.github.io/spider). Lisans CC BY-SA 4.0. |
| Ana ölçüm / ayar | Spider dev (1.034 soru) | Tüm ayar burada yapılır. Hızlı iterasyon için sabit tohumlu ~300 soruluk `dev_small`. |
| Final ölçüm | Spider test (resmi zip içinde, 2024'ten beri açık) | Yalnızca final konfig için **bir kez**. |
| Genelleme | BIRD Mini-Dev, SQLite sürümü (500 soru) | `github.com/bird-bench/mini_dev` / HF `birdsql/bird_mini_dev`. Hiç görmediği, daha kirli veritabanları. BIRD "evidence" ipucu alanı prompt'a dahil edilir (BIRD'in standart kurgusu). |

- **Demo/elle test veritabanları:** Spider'ın 200 veritabanı (özellikle modelin görmediği dev DB'leri) + **Chinook** (açık örnek müzik mağazası DB'si, model hiç görmedi). Kullanıcı herhangi bir `.sqlite` dosyasını da gösterebilir; şema otomatik okunur. Sentetik DB üretilmez.
- **Veritabanı türü: yalnızca SQLite** (kullanıcı kararı). PostgreSQL/MySQL/MSSQL kapsam dışı.
- İndirilen veriler `data/` altında, git dışında; indirme script'i idempotent ve dosya boyutu/sayısı doğrulaması yapar.
- Train ile dev/test arasında veritabanı çakışması yoktur (Spider'ın kurgusu: farklı veritabanları) — bu, ölçümün "yeni şemaya genelleme" olduğu anlamına gelir; README'de belirtilir.

## 6. Akış

```
Spider/BIRD ──► data/ (indirme + doğrulama)
                    │
                    ▼
            prompt builder  ── şema (CREATE TABLE) [+ isteğe bağlı örnek satırlar] + soru
                    │
        ┌───────────┴───────────────┐
        ▼                           ▼
   SFT veri seti               inference (vLLM, batch)
   (chat formatı,              baseline modeller + LoRA adapter'ları
    loss sadece SQL'de)               │
        │                             ▼
        ▼                       SQL çıkarıcı (yanıttan SQL'i ayıkla)
   trainer (Unsloth + TRL)            │
   LoRA / QLoRA, MLflow log           ▼
        │                       evaluator: SQLite'ta gold + tahmin çalıştır,
        ▼                       sonuç kümelerini karşılaştır (EX), test-suite (Spider)
   adapter  ──► merge ──► vLLM servis (Docker Compose)        │
                    └───► GGUF ──► Ollama                    ▼
                                                    MLflow + markdown rapor
```

### 6.1 Modüller

| Paket | Sorumluluk |
|---|---|
| `config/` | pydantic-settings modelleri, YAML yükleyici, `explain-config` |
| `data/` | Spider/BIRD indirme, doğrulama, ortak `Example` şeması (`question`, `db_id`, `db_path`, `gold_sql`, `difficulty`, `evidence?`) |
| `prompting/` | Şema serileştirme (CREATE TABLE; örnek satır sayısı config'den), prompt şablonu, chat formatı. Eğitim ve çıkarım **aynı** kurucuyu kullanır. |
| `training/` | SFT veri seti hazırlama, Unsloth + TRL `SFTTrainer`, LoRA/QLoRA config'i, MLflow loglama, adapter kaydetme |
| `inference/` | Ortak `Backend` arayüzü: `vllm` (birincil), `ollama`, `claude_cli` (`claude -p --model <config>`, her soru bağımsız oturum, Claude Code'un sistem prompt'u ve araçları kapalı — flag'ler kurulumda doğrulanır; eşzamanlılık ve hız sınırı config'den). Paralel/batch istek, SQL çıkarıcı. Tüm backend'ler aynı prompt kurucuyu kullanır. |
| `evaluation/` | SQLite yürütücü (salt okunur, zaman aşımlı), sonuç kümesi karşılaştırıcı, EX, Spider test-suite accuracy (resmi `taoyds/test-suite-sql-eval`), hata türü sınıflandırma, gecikme ölçümü, rapor |
| `export/` | LoRA merge, GGUF dönüşümü (Unsloth `save_pretrained_gguf`), Ollama Modelfile |
| `app/` | Gradio arayüzü: DB seç, şemayı gör, soruyu yaz; seçilen modellerin (örn. baseline, fine-tune, Haiku) SQL'i ve çalıştırılmış sonucu yan yana |
| `notebooks/` | Düz `peft` öğrenme notebook'u + Unsloth hız karşılaştırması |

### 6.2 Değerlendirme kuralları

- **Ana metrik: execution accuracy (EX).** Gold ve tahmin SQL'i aynı SQLite veritabanında çalıştırılır; sonuçlar eşitse doğru. Metin karşılaştırması yapılmaz (aynı sonucu veren farklı yazımlar doğru sayılır).
- **Spider'da ek: test-suite accuracy** (resmi script ve onun dağıttığı varyant veritabanları). EX ile farkı "şans eseri doğru" oranını gösterir.
- Karşılaştırma kuralları: gold'da `ORDER BY` yoksa satır sırası önemsiz (multiset karşılaştırma); kolon sırası, tekrar eden satırlar ve ondalık yuvarlama için tutarlı kurallar — EX tarafında BIRD'in resmi `evaluation.py` davranışı referans alınır.
- Çalışmayan SQL = yanlış, ayrı hata türü. Zaman aşımı (config, varsayılan birkaç saniye) = yanlış, ayrı tür.
- Veritabanları salt okunur açılır (model SQL'i veriyi değiştiremez).
- Gecikme ve VRAM aynı koşullarda ölçülür (aynı sunucu, aynı batch/eşzamanlılık ayarı); ölçüm koşulları raporda yazılır.

### 6.3 Modeller

| Rol | Varsayılan | Not |
|---|---|---|
| Eğitilecek model | **`Qwen/Qwen2.5-Coder-3B-Instruct`** (kullanıcı kararı, 2026-10-23 araştırmasından sonra) | Gerekçe: text-to-SQL literatürünün (FINER-SQL, SLM-SQL, OmniSQL, CSC-SQL) ortak tabanı → sonuçlar kıyaslanabilir; standart transformer, QLoRA sorunsuz; aynı aileden 7B/14B var. Qwen3.5-4B elendi: Unsloth QLoRA'yı önermiyor, hibrit doğrusal-attention mimarisi, vLLM LoRA desteği doğrulanmadı. |
| Büyük karşılaştırma | `Qwen2.5-Coder-7B-Instruct`, `Qwen2.5-Coder-14B-Instruct` | Yalnızca çıkarım; 16 GB'a sığacak şekilde quantize (vLLM için AWQ/FP8, olmazsa Ollama GGUF Q4). |
| Yeni nesil | `Qwen3.5-4B` ve/veya `Qwen3.5-9B`, thinking kapalı | Yalnızca çıkarım. vLLM ≥ 0.17 gerekir. |
| Claude | Claude Haiku (model adı config'den; Sonnet de denenebilir) | `claude_cli` backend'i, tek seferlik (agent/araç yok), aynı prompt. |
| Referans tavan | `griffith-bigdata/FINER-SQL-3B-Spider` | Aynı 3B tabanın SFT + GRPO + distilasyonla ulaştığı yer; kendi prompt formatı gerekebilir, raporda belirtilir. |

## 7. Eğitim

- **Kütüphane:** Unsloth (hız, VRAM) + TRL `SFTTrainer` + HF `peft` adapter formatı. Kullanıcı kararı: asıl deneyler Unsloth ile; düz `transformers + peft + TRL` öğrenme notebook'unda (aynı model kimliği, aynı ayarlar, 50 adımlık hız/VRAM karşılaştırması).
- **Yöntem:** ana deney QLoRA (`load_in_4bit: true`); aynı modelde LoRA (`load_in_4bit: false`) karşılaştırması. QLoRA = LoRA + 4-bit taban; adapter her iki durumda 16-bit.
- **Loss yalnızca cevap (SQL) token'larında**; prompt token'ları maskelenir.
- **Config'den gelen parametreler (açıklamalı):** `r`, `lora_alpha`, `lora_dropout`, `target_modules`, `load_in_4bit`, `learning_rate`, `num_train_epochs`, `per_device_train_batch_size`, `gradient_accumulation_steps`, `warmup_ratio`, `lr_scheduler_type`, `weight_decay`, `max_seq_length`, `seed`, şema örnek satır sayısı.
- MLflow: loss eğrisi, parametreler, süre, tepe VRAM, adapter artifact'i.
- İlk koşu: 50 adımlık duman testi → gerçek adım/saniye ve VRAM ölçülür, süre tahmini güncellenir.

## 8. Deneyler (lean)

Her satır bir YAML; ayar `dev_small`/dev'de, final konfig test'te bir kez.

1. Baseline: 2–3 küçük aday model, prompt ile → model seçimi
2. SFT varsayılan konfig (QLoRA)
3. LoRA vs QLoRA (doğruluk, eğitim süresi, VRAM)
4. `r`: 8 / 16 / 64
5. Epoch: 1 / 2 / 3
6. Şemada örnek satır: 0 / 3
7. Karşılaştırma modelleri: 7B, 14B, Qwen3.5, Claude Haiku, FINER-SQL-3B
7b. Veri miktarı eğrisi: 1.000 / 3.000 / 7.000 eğitim örneği (ucuz, isteğe bağlı)
8. Final: Spider test + BIRD Mini-Dev (tablodaki her satır için bir kez)

**Final konfig nasıl seçilir (Claude'un kararı):** 3–6 numaralı taramalar varsayılan konfigden **tek faktör değiştirilerek** yapılır; ardından her taramanın en iyi değeri birleştirilip bir doğrulama koşusu yapılır. LoRA mu QLoRA mı final olacağına dev EX'e göre karar verilir; diğeri tabloda aynı en iyi hiperparametrelerle yer alır. Toplam ~8–10 eğitim koşusu.

## 9. Servis (deploy)

- **vLLM:** Docker Compose servisi, OpenAI uyumlu API (`/v1/chat/completions`). Taban model + LoRA adapter (`--enable-lora`) ile tek sunucudan baseline ve fine-tune'a istek atılabilir; birleştirilmiş (merged) model ile de servis edilebilir. Değerlendirme de bu sunucu üzerinden batch halinde koşar.
- **Ollama (zaman kalırsa):** birleştirilmiş model GGUF'a çevrilip Modelfile ile yüklenir ("yerel kullanım" adımı).
- **Demo:** Gradio arayüzü (`make app`) — kullanıcı kendi sorusunu yazar, modellerin SQL'leri ve sonuçları yan yana. Terminal için `make ask DB=<db_id> Q="..." MODEL=<ad>`. Model İngilizce verilerle eğitildiği için Türkçe soru davranışı demoda gözlemlenir ve bulgu olarak not edilir.
- Model OpenAI uyumlu olduğu için paperlens'e de takılabilir (bu projede yapılmaz).

## 10. Hata yönetimi

- LLM istekleri: timeout + retry (tenacity); kalıcı hata ayrı sayılır.
- Yanıtta SQL bulunamazsa: "çıkarılamadı" hata türü.
- SQL yürütme: zaman aşımı, salt okunur bağlantı, istisna yakalanır ve türüne göre sınıflanır.
- Eğitim: checkpoint'ler belirli adım aralığında kaydedilir; OOM durumunda raporda net mesaj ve önerilen ayar (batch ↓, `max_seq_length` ↓).

## 11. Test

- **Unit (CI):** prompt kurucu (şema serileştirme), SQL çıkarıcı, sonuç kümesi karşılaştırıcı (elle hazırlanmış eşdeğer/eşdeğer olmayan sorgu çiftleri, küçük örnek SQLite ile), hata sınıflandırma, config doğrulama.
- **Doğruluk testi:** bizim EX'in resmi evaluator ile küçük bir örnek üzerinde aynı sonucu verdiğini doğrulayan test.
- **GPU gerektirenler** (eğitim duman testi, vLLM) yerelde `make` hedefleriyle.

## 12. Ortam

- WSL2 Ubuntu, `uv`, `ruff`, `mypy`, `pytest`, `pre-commit`.
- Python sürümü: Unsloth + vLLM + bitsandbytes'ın birlikte çözüldüğü sürüm (büyük ihtimalle 3.12); kurulumun ilk görevinde `uv lock` ile doğrulanır.
- Eğitim ve vLLM aynı GPU'yu paylaşır; aynı anda çalıştırılmaz (`make` hedefleri bunu belirtir).
- Docker Compose: `vllm`, `mlflow`. Ollama compose dışında (paperlens'teki gibi).
- `Makefile`: `setup`, `download`, `explain-config`, `train`, `eval`, `serve`, `export-gguf`, `ask`, `test`, `lint`.
- GitHub Actions: lint + typecheck + unit testler.

## 13. Aşamalar (dikey dilim sıralaması)

| Aşama | Kapsam | Bitti sayılır |
|---|---|---|
| **M0** (ilk dilim, hedef 1–2 gün) | Ortam doğrulama (GPU/torch/Unsloth/bitsandbytes/Docker GPU), repo iskeleti + config, Spider indirme, prompt kurucu, minimal değerlendirici (EX), vLLM'de 3B baseline (`dev_small`), 50 adımlık duman testi, tam QLoRA koşusu, fine-tune sonrası değerlendirme | Aynı 300 soruda baseline vs QLoRA EX (ilk Δ) |
| **M1** | BIRD Mini-Dev (başlık seti), few-shot + retry satırları, 7B/14B, Claude (kota ölçümü → tekli/batch kararı), eğitim taramaları (LoRA vs QLoRA, `r`, epoch, örnek satır, isteğe bağlı veri miktarı), bench + maliyet + Pareto grafiği, rapor + bootstrap, Spider test-suite, öğrenme notebook'u | Dev tablosu dolu, final konfig seçili |
| **M2** | Anlamsal hata analizi, vLLM Compose servisi, Gradio + Chinook demo + GIF, final test koşuları (her satır bir kez), README, push (onayla) | §2'deki kriterler |
| *Zaman kalırsa* | Qwen3.5-4B/9B, FINER-SQL-3B, GGUF/Ollama, ~30 soruluk güvenlik/robustluk seti (cevaplanamaz, yıkıcı istek, injection, büyük şema), BIRD parity testi, **GRPO** | — |

Kesme sırası (zaman daralırsa): Qwen3.5 → FINER-SQL → BIRD parity → control_40 → GGUF/Ollama → LoRA satırı (QLoRA kalır).

## 14. Kapsam dışı

PostgreSQL/MySQL/MSSQL, BIRD train ile eğitim, Spider 2.0, full fine-tuning, W&B, cloud deploy, Kubernetes, Gradio dışında UI (Next.js vb.), FastAPI katmanı (vLLM'in API'si yeterli), paperlens entegrasyonu.

## 15. Açık riskler

| Risk | Önlem |
|---|---|
| Blackwell + Unsloth/vLLM/bitsandbytes uyumsuzluğu | İlk görev: GPU doğrulama + sürüm sabitleme; vLLM için resmi Docker imajı yedek yol |
| 7B/14B quantize sürümleri vLLM'de 16 GB'a sığmaz ya da yoktur | Ollama GGUF Q4 ile ölçülür; gecikme karşılaştırmasında sunucu farkı raporda belirtilir |
| Fine-tune kazancı küçük çıkar | Bulgu olarak raporlanır; hata türü analizi nedenini gösterir |
| `claude -p` kota/hız sınırı | Küçük alt kümeler, düşük eşzamanlılık, kaldığı yerden devam eden (cache'li) koşucu |
| Spider DB zip'i / test-suite veritabanları indirme sorunları (Google Drive) | İndirme script'inde elle indirme talimatı yedeği |
