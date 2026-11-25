# Xəttat — Plan 1: Core ML pipeline

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Repo skeleton + data pipeline + baselines + char-level neural models (BiLSTM, Transformer) with resumable Slurm training and evaluation, so the team can produce the first results table on the GPU server.

**Architecture:** Diacritic restoration as per-character binary tagging (keep / convert) over candidate letters only. Pure-Python core (`text`, `metrics`, `baselines`, `augment`) has no torch dependency and is unit-tested anywhere; torch code (`models`, `dataset`, `train`, `evaluate`, `restorer`) is tested on the server (tests auto-skip without torch).

**Tech Stack:** Python 3.11, PyTorch 2.x (cu128), HF `datasets` (data download only), PyYAML, pytest, TensorBoard.

**Spec:** `docs/superpowers/specs/2026-11-25-xettat-design.md`

**Later plans (not here):** Plan 2 — ByT5 fine-tune + ablations; Plan 3 — ONNX export, Docker API, Chrome extension, website; Plan 4 — report & slides.

## Global Constraints

- Candidate map exactly: `e→ə i→ı o→ö u→ü c→ç s→ş g→ğ`, `E→Ə I→İ O→Ö U→Ü C→Ç S→Ş G→Ğ`. Label 1 = convert, 0 = keep. Non-candidate chars never change; output length == input length.
- Never use `str.lower()/upper()` on Azerbaijani text; use `az_lower/az_upper` (`I↔ı`, `İ↔i`).
- All text NFC-normalised.
- Split at **document** level, seed-fixed, frozen to files before any experiment; test set used only by `evaluate.py --split test`.
- Peak GPU memory ≤ 20 GB; bf16 autocast; checkpoint every `ckpt_every` steps; training resumes automatically from `<ckpt_dir>/last.pt`.
- Every run writes `runs/<run_id>/{config.yaml,metrics.jsonl}`; final eval writes `runs/<run_id>/final_<split>.json`.
- Secrets: `.gitignore` must contain `team14*`, `README-team14.md`, `*.conf`.
- Seeds: `seed: 42` default, set for `random`, `numpy`, `torch`.

## Review Focus

1. Input that already contains correct Azerbaijani letters (`üzümün suyu uzume`) — those must be kept unchanged (test in Task 1, Task 5).
2. Uppercase / all-caps text (`AZƏRBAYCAN` stripped to `AZERBAYCAN`, sentence-initial `İ`) — must round-trip (Task 1).
3. Non-letter content: digits, URLs, emojis, punctuation, Cyrillic/Latin English words — untouched, no crash (Task 1, Task 5).
4. Very long input (> `max_len`) at inference — must be chunked, not truncated (Task 5).
5. Empty string / whitespace-only — returns as-is (Task 1, Task 5).

---

### Task 1: Repo skeleton + text core

**Files:**
- Create: `README.md`, `requirements.txt`, `.gitignore`, `pyproject.toml`, `src/xettat/__init__.py`, `src/xettat/text.py`
- Test: `tests/test_text.py`

**Interfaces — Produces (`xettat.text`):**
- `TO_AZ: dict[str,str]`, `TO_ASCII: dict[str,str]` (inverse), `CANDIDATES: frozenset[str]`
- `nfc(s: str) -> str`
- `az_lower(s: str) -> str`, `az_upper(s: str) -> str`
- `strip(s: str) -> str` — replace every Azerbaijani letter by its ASCII form.
- `make_labels(target: str) -> tuple[str, list[int]]` — returns `(strip(target), labels)`; label at position i is 1 iff `target[i] in TO_ASCII`, 0 if `strip(target)[i]` is a candidate but unchanged, `-100` for non-candidates.
- `apply_labels(src: str, labels: Sequence[int]) -> str` — converts candidate chars whose label is 1.

- [ ] **Step 1: Write failing tests** in `tests/test_text.py`:
  - `strip("üzümün suyu üzümə sıçradı") == "uzumun suyu uzume sicradi"`
  - `strip("İLHAM ƏLİYEV") == "ILHAM ELIYEV"`; `az_lower("IŞIQ İL") == "ışıq il"`; `az_upper("ışıq il") == "IŞIQ İL"`
  - `src, y = make_labels("Şəhər")` → `src == "Seher"`, `y == [1, 1, -100, 0, -100]`
  - `apply_labels(*make_labels(t)) == t` for a list of 10 varied sentences (digits, URL `https://x.az`, emoji, English word, all-caps)
  - `make_labels("")  == ("", [])`
- [ ] **Step 2:** `pytest tests/test_text.py -q` → FAIL (module missing)
- [ ] **Step 3:** Implement `text.py`; skeleton files (`requirements.txt` pinned: torch==2.7.1, datasets==3.6.0, pyyaml==6.0.2, numpy==2.2.6, tensorboard==2.19.0, pytest==8.4.1, tqdm==4.67.1; README with sections Setup / Data / Train / Reproduce; `.gitignore` per Global Constraints plus `data/raw/ data/processed/ checkpoints/ *.pt *.onnx __pycache__/`).
- [ ] **Step 4:** `pytest tests/test_text.py -q` → PASS
- [ ] **Step 5:** Commit `feat: repo skeleton and Azerbaijani text core`

### Task 2: Data preparation + augmentation

**Files:**
- Create: `src/xettat/data/__init__.py`, `src/xettat/data/clean.py`, `src/xettat/data/prepare.py`, `src/xettat/data/augment.py`, `data/real_test/TEMPLATE.tsv`
- Test: `tests/test_data.py`

**Interfaces — Produces:**
- `clean.split_sentences(text: str) -> list[str]` (regex on `.!?…` + newline; keeps abbreviations like `və s.` reasonable — no external lib)
- `clean.is_good_sentence(s: str, min_chars=15, max_chars=400, min_az_ratio=0.02) -> bool` — rejects too short/long, mostly non-Latin, or with zero Azerbaijani-specific letters over the threshold (Azerbaijani letters / letters).
- `prepare.split_docs(doc_ids: list[str], seed: int, val_frac=0.01, test_frac=0.01) -> dict[str, set[str]]` deterministic (hash-based on `seed:doc_id`).
- `prepare.main(argv)` CLI: `python -m xettat.data.prepare --source wiki --out data/processed --seed 42 [--limit N]` → writes `train.txt val.txt test.txt` (one NFC target sentence per line, deduplicated, test/val sentences removed from train) and `stats.json` (`{"train": n, "val": n, "test": n, "source": ..., "license": "CC BY-SA 4.0"}`). Source `wiki` = HF `wikimedia/wikipedia`, config `20231101.az`. Source `txt:<path>` = local file, one doc per blank-line-separated block (used in tests).
- `augment.make_input(target: str, rng: random.Random, p_partial=0.3, p_keep=0.3, p_upper=0.05) -> str` — returns stripped text; with prob `p_partial` keeps each Azerbaijani letter with prob `p_keep`; with prob `p_upper` uppercases whole sentence via `az_upper` before stripping. Output length == input length.
- `data/real_test/TEMPLATE.tsv` header `input\ttarget` + 3 examples.

- [ ] **Step 1: Failing tests:** sentence splitting on a 3-sentence paragraph; `is_good_sentence` rejects `"ok"` and a Russian sentence, accepts an Azerbaijani one; `split_docs` same output for same seed, disjoint sets, roughly 1%/1% on 10k ids; `prepare.main(["--source","txt:<tmp>","--out",tmp,"--seed","42"])` produces 3 files with no sentence shared across them; `make_input` with `p_partial=0` equals `strip(target)`, with `p_partial=1, p_keep=1` equals target, length always preserved.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3:** implement
- [ ] **Step 4:** `pytest tests/test_data.py -q` → PASS
- [ ] **Step 5:** Commit `feat: data preparation, doc-level split, augmentation`

### Task 3: Metrics + baselines

**Files:**
- Create: `src/xettat/metrics.py`, `src/xettat/baselines.py`, `src/xettat/run_baselines.py`
- Test: `tests/test_metrics.py`, `tests/test_baselines.py`

**Interfaces — Produces:**
- `metrics.score(inputs: list[str], preds: list[str], targets: list[str]) -> dict` with keys: `char_acc` (candidate positions), `word_acc` (words containing ≥1 candidate), `ambig_word_acc` (words whose stripped form maps to ≥2 targets in a provided/derived ambiguity set — pass `ambiguous: set[str] | None`), `sentence_acc`, `wer`, `corruption_rate` (positions where input already had an Azerbaijani letter or a correct ASCII letter and pred differs from target), `per_pair` (`{"e/ə": {"p":..,"r":..,"f1":..}, ...}` for the 7 lowercase pairs, case-folded), `n_sentences`.
- `baselines.Identity().predict(text) -> str`
- `baselines.WordFreq.fit(sentences: Iterable[str]) -> WordFreq`; `.predict(text) -> str` — token-wise (regex `\w+` with Unicode), lookup by `az_lower(strip(word))`, choose most frequent target form, re-apply original casing pattern (lower / Title / UPPER); unknown words unchanged; keeps already-Azerbaijani letters.
- `baselines.Bigram.fit(...)`, `.predict(text)` — for words with ≥2 candidate forms pick argmax `count(prev_form, form)` with backoff to WordFreq.
- `baselines.WordFreq.ambiguous() -> set[str]` — stripped keys with ≥2 forms (used for `ambig_word_acc`).
- `run_baselines.main(argv)`: `python -m xettat.run_baselines --data data/processed --split val|test --out runs/baselines` → writes `runs/baselines/<name>_<split>.json` for identity/wordfreq/bigram.

- [ ] **Step 1: Failing tests:** `score` perfect prediction → all accs 1.0, corruption 0; identity on `"seher"`→`"səhər"` gives `word_acc==0`; WordFreq fitted on `["Səhər gəldi", "səhər tezdən", "şəhər böyükdür"]` predicts `"seher geldi"` → `"səhər gəldi"`, `"SEHER"` → `"SƏHƏR"`, unknown `"xyz"` unchanged; Bigram fitted on `["böyük şəhər", "böyük şəhər", "sabah səhər"]` predicts `"boyuk seher"` → `"böyük şəhər"` and `"sabah seher"` → `"sabah səhər"`.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3:** implement
- [ ] **Step 4:** `pytest tests/test_metrics.py tests/test_baselines.py -q` → PASS
- [ ] **Step 5:** Commit `feat: metrics and dictionary/bigram baselines`

### Task 4: Char models, dataset, resumable training

**Files:**
- Create: `src/xettat/vocab.py`, `src/xettat/models/__init__.py`, `src/xettat/models/char_bilstm.py`, `src/xettat/models/char_transformer.py`, `src/xettat/dataset.py`, `src/xettat/train.py`, `src/xettat/utils.py`, `configs/bilstm.yaml`, `configs/transformer.yaml`, `configs/debug.yaml`
- Test: `tests/test_models.py` (`pytest.importorskip("torch")`)

**Interfaces — Produces:**
- `vocab.CharVocab` — fixed, data-independent: id 0 pad, 1 unk, then ASCII 32–126, Azerbaijani letters, common punctuation `«»–—…’“”`; `encode(s: str) -> list[int]`, `size: int`. Saved/loaded as JSON (`to_json/from_json`).
- `models.build_model(cfg: dict, vocab_size: int) -> nn.Module`; every model: `forward(ids: LongTensor[B,T], mask: BoolTensor[B,T]) -> FloatTensor[B,T]` (logits for label 1).
  - BiLSTM cfg keys: `emb_dim=128, hidden=256, layers=3, dropout=0.2`.
  - Transformer cfg keys: `d_model=256, heads=4, layers=6, ff=1024, dropout=0.1, max_len=512`, learned positions, `nn.TransformerEncoder(batch_first=True, norm_first=True)` (uses SDPA).
- `dataset.SentenceDataset(path, vocab, max_len, augment: bool, seed)` → items `(ids, labels)`; labels from `make_labels`, input built by `augment.make_input` (train) or `strip` (eval); sentences longer than `max_len` are windowed. `dataset.collate(batch) -> (ids, mask, labels)` padded, labels pad `-100`.
- `utils.set_seed(seed)`, `utils.RunLogger(run_dir)` with `.log(dict)` appending JSONL + TensorBoard scalars.
- `train.main(argv)`: `python -m xettat.train --config configs/bilstm.yaml [--run-id NAME] [--max-steps N]`
  - loss: `BCEWithLogitsLoss` on positions with label ≠ -100.
  - AdamW, cosine schedule with warmup, bf16 autocast on CUDA, grad clip 1.0.
  - every `eval_every` steps: val `score` on `val_subset` sentences → log; save `best.pt` on best `word_acc`; early stop after `patience` evals without improvement.
  - every `ckpt_every` steps and on SIGTERM/SIGUSR1: save `last.pt` (model, optim, sched, step, rng states, best metric); on start, resume if `last.pt` exists.
  - ckpt dir: `cfg.ckpt_root/<run_id>/` (default `/data/teams/team14/checkpoints`, overridable by env `XETTAT_CKPT_ROOT`).
- configs: `debug.yaml` (tiny model, `max_steps: 200`, data `tests/fixtures` OK on CPU).

- [ ] **Step 1: Failing tests:** each model returns shape `[2, 17]` on random ids; `collate` pads and masks correctly; `SentenceDataset` item length == label length and labels match `make_labels`; `train.main` with `configs/debug.yaml` on a 200-sentence fixture runs 30 steps on CPU, writes `metrics.jsonl` and `last.pt`; running again with `--max-steps 40` resumes from step 30 (assert logged first step > 30).
- [ ] **Step 2:** run (server or any torch machine) → FAIL
- [ ] **Step 3:** implement
- [ ] **Step 4:** `pytest tests/test_models.py -q` → PASS
- [ ] **Step 5:** Commit `feat: char BiLSTM/Transformer models and resumable training`

### Task 5: Inference + evaluation

**Files:**
- Create: `src/xettat/restorer.py`, `src/xettat/evaluate.py`
- Test: `tests/test_restorer.py` (`importorskip("torch")`)

**Interfaces — Produces:**
- `restorer.Restorer.from_checkpoint(path: str, device="cpu") -> Restorer`; `.restore(text: str, threshold=0.5) -> str`; `.restore_batch(texts: list[str]) -> list[str]`. Splits on lines, windows inputs longer than `max_len` with overlap 64 and takes predictions from window centres; only candidate positions may change; existing Azerbaijani letters kept.
- `evaluate.main(argv)`: `python -m xettat.evaluate --ckpt PATH --data data/processed --split val|test|real --out runs/<run_id>` → `final_<split>.json` = `score(...)` + `{"latency_ms_per_sentence_cpu": ..., "params": ..., "ckpt_mb": ...}`. `--split real` reads `data/real_test/real_test.tsv`.

- [ ] **Step 1: Failing tests:** with an untrained tiny model `restore` returns same length, keeps `"ə"`, leaves `"https://x.az 123 😀"` unchanged, `restore("")==""`; a 3000-char input returns 3000 chars; `evaluate.main` on fixture writes JSON with keys `word_acc` and `latency_ms_per_sentence_cpu`.
- [ ] **Step 2:** run → FAIL
- [ ] **Step 3:** implement
- [ ] **Step 4:** `pytest -q` → PASS (all)
- [ ] **Step 5:** Commit `feat: restorer inference and evaluation`

### Task 6: Server tooling (Slurm + docs)

**Files:**
- Create: `scripts/setup_server.sh`, `slurm/prepare_data.sbatch`, `slurm/train.sbatch`, `slurm/evaluate.sbatch`, `slurm/test_gpu.sbatch`, `scripts/reproduce.sh`, `docs/SERVER_GUIDE_AZ.md`
- Test: `bash -n` on every script; `tests/test_configs.py` (all `configs/*.yaml` load and have required keys).

**Contents decided:**
- `setup_server.sh`: `pip install torch --index-url https://download.pytorch.org/whl/cu128 && pip install -r requirements.txt && pip install -e .`
- `prepare_data.sbatch`: no GPU (`--cpus-per-task=4 --mem=16G --time=02:00:00`), runs `python -m xettat.data.prepare --source wiki --out data/processed --seed 42`, then `python -m xettat.run_baselines --split val`.
- `train.sbatch`: `--gres=gpu:2g.20gb:1 --time=08:00:00 --signal=B:USR1@300`, takes config path as `$1`, run id `$2`; traps USR1 to let `train.py` checkpoint; on exit code 99 (time-limit reached, not finished) does `sbatch --dependency=afterany:$SLURM_JOB_ID` self-resubmit.
- `test_gpu.sbatch`: debug slice `1g.20gb`, 10 min, runs `pytest -q` + `nvidia-smi -L`.
- `reproduce.sh`: prepare data (if missing) → baselines on test → `evaluate --split test` for each checkpoint listed in `runs/HEADLINE` → `python -m xettat.report_table` (writes `report/tables/results.md`). Create `src/xettat/report_table.py` (`main()` collects `runs/*/final_test.json` into a markdown table).
- `SERVER_GUIDE_AZ.md`: step-by-step in Azerbaijani (VPN, clone, setup, submit, monitor with `squeue --me`/`tail -f`, copy results, push `runs/`).

- [ ] **Step 1:** write `tests/test_configs.py` → FAIL
- [ ] **Step 2:** write scripts, configs pass; `bash -n slurm/*.sbatch scripts/*.sh` clean
- [ ] **Step 3:** `pytest -q` → PASS
- [ ] **Step 4:** Commit `chore: slurm scripts, reproduce script, server guide`
