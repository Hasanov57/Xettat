# Xəttat — results & decisions log

Running record for the final report (IEEE) and slides. Numbers copied from server outputs
(`runs/*/final_*.json`, `runs/*/metrics.jsonl`, `data/*/stats.json`). Newest entries at the bottom of each section.

## Data

| Source | Docs | Train sent. | Val | Test | Notes |
|---|---|---|---|---|---|
| Azerbaijani Wikipedia (`wikimedia/wikipedia` 20231101.az, CC BY-SA 4.0) | 196 158 | 3 019 454 | 34 476 | 36 890 | article-level split, seed 42 |
| HPLT 2.0 cleaned `azj_Latn`, shards 0–2 (CC0) | 358 505 | 3 000 000 (capped) | 85 748 | 100 626 | 1 768 docs (0.5%) dropped: az_ratio < 0.08 |
| OpenSubtitles v2024 mono `az` (OPUS) | 698 chunks × 200 lines | 80 932 | 1 879 | 1 660 | **537/698 chunks (77%) dropped as typed without Azerbaijani letters** — evidence of the problem; val/test may share films with train (chunk split) |
| Mix (wiki + HPLT + subs) | – | 5 867 532 + subs | 4 000 + subs | per-domain | ~136k HPLT sentences removed: duplicates of Wikipedia / held-out (web copies of Wikipedia → leakage guard) |

Old orthography (`ä`, Cyrillic `ә`) rejected; documents with az_ratio < 0.08 rejected (normal text ≈ 0.12–0.18).

## Main results — v1 scheme (binary keep/convert), validation sets

Word acc = words containing ≥1 candidate letter restored exactly; Sent = whole sentence exact.

| Model | Params | Wiki word / sent | HPLT web word / sent | Subtitles word / sent | CPU ms/sent (1 thread, torch) |
|---|---|---|---|---|---|
| Identity | – | 31.99 / 0.0 | 29.78 / 0.0 | 32.07 / 0.0 | – |
| Word-frequency lexicon | – | 95.75 / 66.6 | 96.38 / 69.4 | 91.49 / 66.8 | 0.1 |
| Lexicon + bigram | – | 96.48 / 71.5 | 96.80 / 72.6 | 91.67 / 67.6 | 0.1 |
| BiLSTM (wiki) `bilstm_v1` | 3.96M | **98.84 / 89.5** | 98.77 / 88.9 | 98.36 / 92.7 | 43 |
| Transformer (wiki) `transformer_v1` | ~5M | 98.67 / 88.1 | – | – | 21 |
| ByT5-small enc. (wiki) `byt5_v1` | 217.7M | 98.72 / 88.6 | – | – | **646** |
| BiLSTM (mix) `bilstm_mix` | 3.96M | 98.71 / 88.8 | **99.00 / 91.1** | **99.08 / 95.9** | 47 |

Findings:
- Neural models cut word errors ~3× vs the best classical baseline (3.52% → 1.16% on wiki).
- Adding web + dialogue data: −44% word errors on subtitles (1.64→0.92), −19% on web, −0.13 pt on wiki.
- ByT5: train-time val (first 2 000 val sentences) said 99.10, full val 98.72 → subset not representative;
  pretrained 218M model did not beat a 4M task-specific BiLSTM and is 15× slower on CPU. **TODO before report:
  re-check subset vs full-val gap (e.g. evaluate bilstm_v1 on the same first 2 000 sentences).**
- Real-use example fixed by mix data: `xatirlasan` → v1 `xatırlaşan` (wrong, a real word), mix `xatırlasan` ✓;
  `necesen` → v1 `neçəsən`, mix `necəsən` ✓. `agacin ucunda uc alma` → `ağacın ucunda üç alma` (both ✓).

## Writing-style robustness (data/variants/val_*.tsv, 5 000 sentences each)

| Model | std | digraph (sh/ch/gh) | a (ə→a) | mixed | english (word / en-corruption) |
|---|---|---|---|---|---|
| `bilstm_mix` (v1) | 98.67 / 88.7 | 81.61 / 20.5 | 58.52 / 7.0 | 78.44 / 19.3 | 97.78 / **8.8% English words corrupted** |
| `bilstm_v2` (3-class, a_bias 0) | 98.54 / 87.4 | **98.83 / 89.7** | **97.92 / 82.6** | **98.39 / 86.0** | 98.69 / **0.03%** |

`bilstm_v2` train-time val (mixed styles, 6 000 sent.): step 2k → 96.63; 4k → 97.39; 24k → 98.17; 100k → 98.76
(best 98.78 @ 96k). a_false_conversion 0.47% → 0.14%. English corruption 0.0%. HPLT val: 98.89 / 89.95 (mix v1: 99.00).

Findings:
- v2 costs −0.13 pt on standard style, gains +17 (digraph), +39 (a-style), +20 (mixed); English corruption 8.8% → 0.03%.
- Digraph style is *easier* than standard (98.83 vs 98.54): writing `sh` disambiguates ş, less guessing.
- `İshaq` (genuine s+h) kept, `sheher`→`şəhər`, `adad`→`ədəd` while `ata və ana` untouched (examples).

a_bias sweep (int8 ONNX, 2 000 sent./style, word acc / a false-conversion):

| a_bias | std | digraph | a | mixed | english |
|---|---|---|---|---|---|
| 0.0 | 98.65 / 0.10% | 98.90 / 0.08% | 97.95 / 0.46% | 98.45 / 0.27% | 98.79 / 0.10% |
| −1.0 | 98.65 / 0.08% | 98.91 / 0.07% | 97.82 / 0.33% | 98.37 / 0.19% | 98.80 / 0.06% |
| **−1.5 (chosen)** | 98.66 / 0.05% | 98.91 / 0.05% | 97.79 / 0.29% | 98.34 / 0.17% | 98.81 / 0.05% |
| −3.0 | 98.68 / 0.03% | 98.92 / 0.03% | 97.36 / 0.22% | 98.13 / 0.10% | 98.82 / 0.02% |

Rationale: corrupting correctly typed text is the anti-goal and most users write standard style.

## Deployment (Track 3)

| Model | ONNX fp32 | ONNX int8 | int8 decision agreement | Used in extension |
|---|---|---|---|---|
| `bilstm_v1` | 15.85 MB | 4.01 MB | 99.94% | int8 |
| `bilstm_mix` | 15.85 MB | 4.01 MB | 99.70% (< 99.9% gate) | fp32 |
| `bilstm_v2` | 15.85 MB | 4.01 MB | 99.95% | **int8 (current extension)** |

onnxruntime-web 1.20.1 WASM ≈ 11.2 MB. Extension runs fully offline (offscreen document).
ONNX Runtime CPU (int8, 1 thread, Python): **4.6 ms/sentence** for bilstm_v2 (vs 43 ms PyTorch).
TODO: in-browser timing (wasm) on a real laptop; Docker API latency.

## Compute log (GPU = 2g.20gb MIG slice)

| Job | Wall | Notes |
|---|---|---|
| bilstm_v1 | 3h02 | 60k steps |
| transformer_v1 | 3h09 | 80k steps |
| byt5_v1 | 7h45 + 2h40 | OOM at batch 32 without grad-ckpt (fixed); auto-resubmitted at time budget, resumed |
| bilstm_mix | 5h04 | 100k steps (best @ 92k) |
| bilstm_v2 | 5h09 | 100k steps (best @ 96k) |
| variant evals | 2 × ~9 min | |
Total so far ≈ 27.5 GPU-h of 145.

## Engineering incidents (honest "what went wrong")

- Resume bug: RNG state restored on GPU → `TypeError` (caught by unit test on server, fixed before long runs).
- ONNX export: BiLSTM dropped unused `mask` input; Transformer MHA baked sequence length → equivalent functional export path + equivalence test.
- ByT5 OOM on 20 GB → gradient checkpointing (6.5 GB peak).
- OpenSubtitles v2018 URL 404 → v2024.
- 3/5000 variant-TSV rows had TABs inside sentences → mis-paired in evaluation (negligible effect); fixed in generator + loader with a regression test.
- Extension: `Alt+Shift` is the Windows layout switch → shortcut changed to `Ctrl+Shift+Space`; editor iframes → all-frames injection + popup fallback. Open: some online-notepad editor not recognised as editable (copies to clipboard instead).

## Decisions (with user)

- Track 3 (Industry product); extension is the main product, website for info/demo, Docker API for the requirement.
- General spelling/typo correction = out of scope (future work: suggestion-only mode).
- v2 scheme: sh/ch/gh, a→ə, protected URLs/e-mails/@/#, English-word insertion in training.
- Report and UI redesign + Vercel deployment at the end.
