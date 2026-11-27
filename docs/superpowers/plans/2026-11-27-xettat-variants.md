# Xəttat — Plan 3: spelling variants (v2 scheme) + protected spans

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:executing-plans. Steps use `- [ ]`.

**Goal:** 3-class tagging (keep/convert/delete) so `sh/ch/gh` and `a→ə` are restored; rule-protected URLs/emails/@/#; per-style evaluation.
**Spec:** `docs/superpowers/specs/2026-11-25-xettat-design.md` §14.

## Global Constraints
- KEEP=0, CONVERT=1, DELETE=2, IGNORE=-100. v1 (binary) checkpoints/ONNX keep working unchanged.
- DELETE only for `h/H` whose previous *input* char is `s/S/c/C/g/G`; CONVERT only for chars in `TO_AZ_EXT`; masked at inference.
- Protected spans regex identical in Python and JS (tests assert same spans on shared cases).

## Review Focus
1. Genuine `sh` (`İshaq`, `məshəf`) — must be learnable (labelled KEEP), never forced by rule.
2. `a` false conversions (`ata` → `ətə`) — measured by `a_false_conversion_rate`.
3. Uppercase digraphs (`SHEHER`, `Sheher`) — generator produces them, apply handles them.
4. Text with URLs/emails/mentions — untouched in both Python and JS.
5. Window boundaries splitting `s|h` — deletion decided from the window where the `h` is most central; apply uses *input* neighbour, not window neighbour.

### Task 1: text core v2 — `TO_AZ_EXT`, `apply_actions`, `protected_mask`, `make_variant_pair` (pure Python, tests)
### Task 2: metrics — misaligned sentences scored by token alignment; `a_false_conversion_rate`, `english_corruption_rate`
### Task 3: variant eval sets — `python -m xettat.data.variants` → `data/variants/{split}_{style}.tsv`; `evaluate --tsv`
### Task 4: models/dataset/train/infer — `num_classes` (1|3), CE loss, masked argmax, protected spans in `predict_texts`; configs `bilstm_v2.yaml`
### Task 5: export/JS — `scheme` in meta, JS `restoreWithActions` + protected spans, ORT runner handles [B,T,3]
