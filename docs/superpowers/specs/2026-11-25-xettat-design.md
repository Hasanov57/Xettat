# Xəttat — Dizayn sənədi

**Tarix:** 2026-11-25 · **Kurs:** DLE-AI-202 Deep Learning Final Project, Cohort II 2026 · **Track:** 3 — Industry Product
**Repo:** https://github.com/Hasanov57/Xettat · **Code freeze:** 18 dekabr 2026, 23:59 · **Müdafiə:** 20 dekabr 2026

---

## 1. Niyə və nə

Azərbaycanda çox insan ingilis klaviaturası ilə yazır və xüsusi hərfləri əvəz edir:
`ə→e, ı→i, ö→o, ü→u, ç→c, ş→s, ğ→g` (böyük hərflərdə `Ə→E, İ→I, Ö→O, Ü→U, Ç→C, Ş→S, Ğ→G`).
Nəticədə mətn orfoqrafik cəhətdən səhv olur: `uzumun suyu uzume sicradi` → düzgünü `üzümün suyu üzümə sıçradı`.

**Xəttat** — kontekstə baxaraq bu hərfləri bərpa edən, bizim train etdiyimiz model və onun üstündə qurulmuş məhsuldur:
- **Əsas məhsul:** Chrome extension — istənilən saytda (Gmail, WhatsApp Web, Facebook, …) mətn qutusunda bir düymə/qısayolla mətni düzəldir.
- **İkinci üz:** veb sayt — Xəttat haqqında məlumat + canlı demo (mətni yapışdır → düzəlt).
- **Track 3 tələbi:** Docker-də işləyən API, CPU-da latency / model ölçüsü / yaddaş ölçümləri.

**Uğur nədir:** real istifadəçinin yazdığı mətndə sözlərin ≥97% düzgün bərpa olunması, extension-da hiss olunmayan gecikmə (<100 ms orta cümlə üçün, CPU), müdafiədə canlı işləyən demo.

## 2. Məqsədlər, qeyri-məqsədlər, anti-məqsədlər

**Məqsədlər**
- Hərf-hərf diakritik bərpası (yuxarıdakı 7+7 cüt), kontekstə həssas (`seher` → `səhər` / `şəhər`).
- Bir neçə model ailəsinin ədalətli müqayisəsi (baseline → kiçik neyron → pretrained fine-tune).
- Extension (Manifest V3) + sayt + Docker API.
- Kursun reproducibility tələbləri: sabit seed, pinned mühit, run logları, bir əmrlə əsas nəticələrin təkrarı.

**Qeyri-məqsədlər (v1-də yoxdur)**
- `sh/ch/gh` kimi çox-hərfli transliterasiya (`shekil` → `şəkil`). Vaxt qalsa stretch goal.
- Ümumi orfoqrafiya / qrammatika düzəlişi (yanlış yazılmış söz, durğu işarələri).
- Kiril əlifbası, digər dillər.

**Anti-məqsədlər (texniki işləsə də uğursuzluq sayılır)**
- Modelin **düzgün** yazılmış hissəni korlaması (istifadəçi artıq `ə` yazıbsa, toxunmamalıyıq).
- Mətnin uzunluğunu/sözləri dəyişməsi, söz "uydurması" — model yalnız icazəli hərfləri dəyişə bilər.
- İstifadəçinin mətnini icazəsiz kənar serverə göndərmək (extension default olaraq lokalda işləyir).
- Test setinə baxaraq tuning (kurs qaydası).

## 3. Məhdudiyyətlər

- **GPU:** Hyperstack A100-ün 20 GB MIG slice-ı (`2g.20gb`), Slurm ilə (`sbatch`/`srun`). Bir anda 1 job, max 8 saat/job, cəmi **145 GPU-saat**. Notebook GPU görmür, 6 GB RAM.
- **Server:** VPN (WireGuard) arxasında, CUDA 12.8 (`pip install torch --index-url https://download.pytorch.org/whl/cu128`). Disk 180 GB, backup yoxdur → kod GitHub-da, nəticələr kənara kopyalanır.
- **Vaxt:** ~3 həftə (25 noyabr → 18 dekabr).
- **Məxfilik:** `README-team14.md` / `team14.conf` VPN açarı və token saxlayır — **heç vaxt repoya düşməməlidir** (`.gitignore`).
- **Deploy:** CPU və ya bir consumer GPU-da işləməlidir (Track 3).

## 4. Tapşırığın formulası (əsas texniki qərar)

Hərfləri silmək mətnin uzunluğunu dəyişmir: hər çıxış hərfi bir giriş hərfinə uyğun gəlir. Ona görə tapşırıq **hərf səviyyəli ardıcıllıq etiketləməsi** (sequence labelling) kimi qurulur:

- Hər **namizəd hərf** (`e i o u c s g E I O U C S G`) üçün binar qərar: *olduğu kimi saxla* (0) və ya *Azərbaycan hərfinə çevir* (1).
  - `e→ə, i→ı, o→ö, u→ü, c→ç, s→ş, g→ğ`; `E→Ə, I→İ, O→Ö, U→Ü, C→Ç, S→Ş, G→Ğ`.
  - Qeyd: böyük `I` həm `I` (ı-nın böyüyü), həm `İ` ola bilər — `1 = İ` kimi kodlanır.
- Namizəd olmayan hərflər (o cümlədən artıq düzgün yazılmış `ə, ş, …`) **heç vaxt dəyişmir** → "uydurma" mümkün deyil, anti-məqsəd struktur olaraq qorunur.
- Loss yalnız namizəd mövqelərdə hesablanır (mask).

**Diqqət:** Azərbaycan dilində böyük/kiçik hərf çevrilməsi (`I/ı`, `İ/i`) Python-un `str.lower()` ilə səhvdir → öz `az_lower/az_upper` funksiyalarımız.

## 5. Data

**Prinsip:** düzgün yazılmış Azərbaycan mətni götür → diakritikləri sil → (giriş, hədəf) cütü. Limitsiz, pulsuz, etiketləmə lazım deyil.

| Mənbə | Təxmini həcm | Lisenziya | Rol |
|---|---|---|---|
| Azərbaycan Wikipedia (HF `wikimedia/wikipedia`, `20231101.az`) | ~200k məqalə | CC BY-SA 4.0 | əsas train/val/test |
| CC-100 / HPLT / OSCAR az (biri, mövcudluğa görə) | on milyonlarla cümlə | ToS/CC0 (mənbəyə görə qeyd) | daha çox və daha "danışıq" üslub |
| **Real istifadəçi test seti** (komanda yazır) | 500+ cümlə | bizim | final "real dünya" qiymətləndirməsi |

**Emal (pipeline `src/xettat/data/`):**
1. Cümlələrə bölmək, NFC normallaşdırma, qısa (<15 hərf) / çox uzun / qeyri-Azərbaycan (xüsusi hərf nisbəti aşağı) cümlələri atmaq, dublikatları silmək.
2. Split **sənəd (məqalə) səviyyəsində**: train 98% / val 1% / test 1% (test ≥ 20k cümlə). Split seed ilə sabitlənir və fayl kimi saxlanılır — eksperimentlərdən əvvəl donur.
3. **Noise augmentasiyası (yalnız train):** real istifadəçilər bəzən bəzi hərfləri düzgün yazır → cümlələrin ~30%-də diakritiklərin təsadüfi hissəsi saxlanılır; həmçinin təsadüfi böyük hərf / bütün sözün böyük yazılması.
4. **Real test seti:** komanda üzvləri gündəlik yazdıqları kimi (WhatsApp üslubu) ASCII-də cümlələr yazır + düzgün versiyası. Toplama şablonu repoda olacaq (`data/real_test/TEMPLATE.tsv`). Bu set yalnız finalda istifadə olunur.

Kursun minimumu (≥10k etiketli nümunə) dəfələrlə ötülür; split ölçüləri hesabatda verilir.

## 6. Yanaşmalar (müqayisə olunan model ailələri)

Hamısı eyni split, eyni metriklər ilə.

| # | Model | Nə edir | Gözlənti / rol |
|---|---|---|---|
| B0 | **Identity** | heç nə dəyişmir | aşağı sərhəd |
| B1 | **Söz-tezlik lüğəti** | hər ASCII sözü train-də ən çox görülən düzgün formasına çevirir; tanımadığını saxlayır | kursun "sadə baseline"-ı, kontekst görmür |
| B2 | **Lüğət + bigram kontekst** | qeyri-müəyyən sözlərdə qonşu sözə görə seçim | güclü klassik baseline (ədalətli müqayisə üçün) |
| M1 | **Char BiLSTM** (scratch) | hərf embedding → 2–3 qat BiLSTM → binar baş | kiçik (~2–5 MB), çox sürətli; extension-un default modeli üçün namizəd |
| M2 | **Char Transformer encoder** (scratch) | 4–6 qat, lokal+qlobal kontekst | M1 ilə eyni ölçü sinfində alternativ arxitektura |
| M3 | **ByT5-small encoder fine-tune** (pretrained, ~300M → yalnız encoder ~ 1/2) | bayt-səviyyəli pretrained encoder + binar token-classification başı | ən dəqiq olması gözlənilir; Docker API-də "keyfiyyət" rejimi |
| (stretch) M4 | **Distillasiya** M3 → M1/M2 | kiçik model böyüyün proqnozlarından öyrənir | extension-da keyfiyyət/sürət kompromisi |

**Niyə seq2seq (tam generasiya) deyil:** daha yavaş, uzunluğu/sözləri dəyişə bilər (anti-məqsəd). Tagging formulası eyni məlumatı daha təhlükəsiz və sürətli verir. ByT5 bayt-səviyyəli olduğu üçün ASCII giriş 1 hərf = 1 token, uyğunlaşdırma trivialdır (ASCII olmayan hərflər çox-bayt olur, onlar namizəd deyil).

**Ablation/analiz (kursun rigor hissəsi):** augmentasiyalı vs augmentasiyasız train; kontekst pəncərəsinin uzunluğu; model ölçüsü vs dəqiqlik; train data həcmi (10%/30%/100%) əyrisi; hərf cütlərinə görə xəta (gözlənti: `e/ə` və `i/ı` ən çətin).

## 7. Metriklər

- **Hərf dəqiqliyi** (yalnız namizəd mövqelərdə), hərf cütünə görə precision/recall/F1.
- **Söz dəqiqliyi / WER** (əsas metrik) — həm bütün sözlər, həm yalnız qeyri-müəyyən sözlər üzrə.
- **Cümlə dəqiqliyi** (bütün cümlə tam düzgün).
- **"Korlama dərəcəsi":** düzgün olan hərfin səhvə çevrilmə tezliyi (anti-məqsəd üçün).
- **Deploy (Track 3):** CPU latency (p50/p95, cümlə və 1000 hərf), throughput, model faylı ölçüsü (MB), RAM peak; brauzerdə (onnxruntime-web) latency.

Bütün nəticələr Wikipedia test + real istifadəçi test setində ayrı-ayrı verilir.

## 8. Training və eksperiment infrastrukturu

- **Kod:** Python 3.11, PyTorch (cu128), HF `transformers`/`datasets` (yalnız M3 və data yükləmə üçün).
- **Konfiq:** hər eksperiment `configs/*.yaml`; seed, data versiyası, hiperparametrlər orda.
- **Resumable training:** hər N addımda checkpoint (`/data/teams/team14/checkpoints/<run>/`), job yenidən işə düşəndə son checkpoint-dən davam edir; early stopping val söz-dəqiqliyinə görə. bf16 AMP.
- **Tracking:** hər run üçün `runs/<run_id>/` — `config.yaml`, `metrics.jsonl` (train/val), `final_test.json`, TensorBoard logları. Kiçik fayllar git-ə commit olunur (kursun "exported logs" tələbi). Böyük checkpoint-lər git-də deyil.
- **Slurm:** `slurm/*.sbatch` hazır skriptlər (data hazırlığı CPU-da, training `2g.20gb`-da, `--time` realistik). Debug üçün `1g.20gb` + kiçik subset.
- **GPU büdcəsi (təxmini):** data ~0 (CPU), M1 ~3 saat, M2 ~4 saat, M3 ~25 saat, ablationlar ~30 saat, ehtiyat ~80 saat → 145 saat çərçivəsində.
- **Bir əmrlə təkrar:** `bash scripts/reproduce.sh` — data hazırlığı → baseline-lar → saxlanmış checkpoint-lərlə test → `report/tables/results.md` və plotlar. (Tam yenidən train ayrıca `--retrain` bayrağı ilə.)

## 9. Məhsul

**Export:** M1/M2 → ONNX (və int8 kvantlaşdırılmış variant). M3 → ONNX (və ya PyTorch) API daxilində.

**Chrome extension (`extension/`, Manifest V3) — əsas məhsul**
- Model **brauzerin içində** işləyir (`onnxruntime-web`, WASM), şəbəkə/server lazım deyil, mətn kompüterdən çıxmır.
- İstifadə: istənilən `textarea` / `input` / `contenteditable` sahədə qısayol (`Alt+Shift+X`, dəyişdirilə bilər) və ya sahənin yanında kiçik "Xəttat" düyməsi → bütün mətn (və ya seçilmiş hissə) düzəlir. Kontekst menyu: "Xəttat ilə düzəlt". `Ctrl+Z` ilə geri qaytarmaq işləməlidir (`execCommand('insertText')` yolu ilə).
- Popup: mətni yapışdır-düzəlt sahəsi, qısa təlimat, seçim: "lokal model" / "server (daha dəqiq)" — server rejimində API URL konfiqurasiya olunur.
- Çətin saytlar (Google Docs, bəzi zəngin editorlar) üçün ən azı popup/kontekst menyu yolu işləməlidir; tam inteqrasiya "best effort".

**Veb sayt (`web/`)** — statik səhifə (GitHub Pages): Xəttat nədir, necə işləyir, canlı demo (eyni ONNX model brauzerdə), extension-u quraşdırma təlimatı, komanda.

**Docker API (`serve/`)** — FastAPI: `POST /restore {"text": "..."} → {"text": "...", "changes": [...]}`, `GET /health`; `model` parametri (`fast` = M1/M2, `accurate` = M3). `Dockerfile` CPU üçün; `docker run -p 8000:8000 xettat` ilə işləyir. Latency ölçmə skripti `scripts/benchmark.py`.

## 10. Repo strukturu (kursun tövsiyəsinə uyğun)

```
Xettat/
├── README.md              # setup, data, bir əmrlə reproduction
├── requirements.txt       # pinned
├── .gitignore             # data, checkpoint, team14 sirləri
├── configs/               # eksperiment yaml-ları
├── src/xettat/            # data/, models/, train.py, evaluate.py, baselines.py, export.py
├── slurm/                 # sbatch skriptləri
├── scripts/               # reproduce.sh, benchmark.py, setup_server.sh
├── runs/                  # izlənən run logları
├── serve/ + Dockerfile    # API
├── extension/             # Chrome extension
├── web/                   # sayt
├── data/real_test/        # real istifadəçi test şablonu (data özü yüklənir)
├── report/                # report.tex, report.pdf (IEEE)
├── presentation/          # presentation.pdf
└── docs/                  # bu sənəd, planlar, server təlimatı
```

## 11. Rollar (komanda dəqiqləşdirəcək)

Kurs hər modulun sahibini tələb edir: **Data**, **Modelləşdirmə**, **Eksperimentlər/qiymətləndirmə**, **Deploy (extension/sayt/API)**, **Hesabat/təqdimat**. Komanda adları sonra əlavə olunacaq.

## 12. Təqvim

| Tarix | Mərhələ |
|---|---|
| 25–28 noy | Repo skeleti, data pipeline, baseline-lar, server quraşdırması, real test toplanmağa başlayır |
| 29 noy – 4 dek | M1, M2 train; ilk extension prototipi (M1 ilə) |
| 5–10 dek | M3 fine-tune, ablationlar, ONNX export, Docker API |
| 11–15 dek | Extension/sayt cilalanması, real test qiymətləndirməsi, benchmark |
| 15–18 dek | Hesabat (IEEE), slaydlar, `v1.0-final` tag |
| 20 dek | Müdafiə |

## 13. Qərarlar və builder-ə qalanlar

**İstifadəçi qərarları:** Track 3; extension əsas, sayt təqdimat/məlumat üçün; işi Claude qurur, GPU əmrlərini komanda işə salır.

**Claude-un qərarları (düzəldilə bilər):** tagging formulası; model siyahısı (B0–M3); extension-da lokal (brauzerdə) inference default; Wikipedia + bir böyük web korpus; v1-də `sh/ch/gh` yoxdur; qısayol `Alt+Shift+X`; tracking üçün JSONL + TensorBoard (xarici hesab tələb etmir).

**Builder-ə qalanlar:** dəqiq hiperparametrlər, cümlə bölücü detalları, extension UI-nin vizual detalları, web korpusun konkret seçimi (yükləmə mövcudluğuna görə).
