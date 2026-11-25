# Xəttat — Plan 2: ByT5 model + product (ONNX, API, extension, website)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add the pretrained ByT5 tagger (M3) and everything needed to ship: ONNX export of char models, a JS inference core shared by the Chrome extension and the website, the MV3 extension, the static website, and a Dockerised FastAPI service.

**Architecture:** Python side adds `byt5` to `build_model` (same `forward(ids, mask)` contract over CharVocab ids; bytes are expanded inside the model) and `export.py` (char models → ONNX + `vocab.json` + `meta.json`). Browser side: one dependency-free ES module `web/shared/xettat-core.js` (encode, windows, apply) + a thin ORT runner; the extension runs ORT inside an **offscreen document**; the website reuses the same files. The API wraps `Restorer`.

**Tech Stack:** PyTorch, HF `transformers` (ByT5), `onnx`, FastAPI + uvicorn, onnxruntime-web 1.20 (WASM), Chrome MV3, plain HTML/CSS/JS (no build step), `node --test` for JS tests.

**Spec:** `docs/superpowers/specs/2026-11-25-xettat-design.md` (§6 M3, §9 product)

## Global Constraints

- All Plan 1 constraints hold (candidate map, length preservation, only candidates change, NFC).
- ONNX model I/O: inputs `ids` int64 [B,T], `mask` bool [B,T]; output `logits` float32 [B,T]; dynamic B and T; opset 17.
- JS core must produce byte-identical output to `xettat.infer.predict_texts` for the same probabilities (same windowing: overlap 64, centre-preference; threshold 0.5).
- Extension: Manifest V3, permissions limited to `storage`, `contextMenus`, `offscreen`, `activeTab`, `scripting`; no remote code; text never leaves the browser in local mode.
- Default shortcut `Alt+Shift+X`; context-menu item "Xəttat ilə düzəlt".
- API: `POST /restore {"text": str, "model": "fast"|"accurate"} → {"text": str, "changed": int, "ms": float}`, `GET /health`; max 20 000 chars per request (413 above).
- ByT5: `google/byt5-small` encoder only + linear head; ≤ 20 GB peak (bf16, batch ≤ 32, max_len 512 chars).

## Review Focus

1. Pages whose editor is `contenteditable` (Gmail, WhatsApp Web) — replacement must go through `document.execCommand('insertText')` so Undo works and the site's own state updates.
2. Text selected inside an input — only the selection is restored.
3. Model files missing from the extension (ORT not fetched) — popup shows a clear message instead of failing silently.
4. API given > 20 000 chars or non-JSON — 413/422, not 500.
5. ByT5 with already-typed Azerbaijani letters (multi-byte) — alignment stays correct.

---

### Task 1: ByT5 tagger
Files: `src/xettat/models/byt5.py`, `models/__init__.py` (+`byt5`), `configs/byt5.yaml`, `tests/test_byt5.py` (importorskip transformers).
Produces: `ByT5Tagger(vocab_size, name="google/byt5-small", dropout=0.1, pretrained=True)`; `forward(ids, mask) -> [B,T]`. Char→bytes expansion vectorised via lookup tables (`byte_len`, `byte_ids[char_id, 4]`, ids = byte+3), gather hidden state at each char's first byte.
Tests: output shape `[2,T]` for a random tiny config (`pretrained=False`, tiny `T5Config`); a string with `ə` gives same logits length as chars.

### Task 2: ONNX export + CPU benchmark
Files: `src/xettat/export.py`, `scripts/benchmark.py`, `tests/test_export.py` (importorskip onnx, onnxruntime).
Produces: `python -m xettat.export --ckpt best.pt --out exported/<name>/` → `model.onnx`, `model.int8.onnx` (dynamic quantisation, if onnxruntime.quantization available), `vocab.json`, `meta.json {"max_len", "threshold", "model_type", "params", "source_ckpt"}`. Test: exported tiny model logits match torch within 1e-4. `benchmark.py` prints p50/p95 ms per sentence + file sizes + peak RSS.

### Task 3: JS core (shared)
Files: `web/shared/xettat-core.js`, `web/shared/xettat-ort.js`, `web/tests/core.test.mjs`.
Produces (`xettat-core.js`, ES module, no deps): `CANDIDATES`, `TO_AZ`, `makeVocab(chars) -> {encode(str): Int32Array-like number[]}`, `windows(n, maxLen, overlap) -> [s,e][]`, `restoreWithProbs(text, probsFn, maxLen) -> string` where `probsFn(chunks: string[]) -> Promise<Float32Array[]>`, `splitLines` handling. `xettat-ort.js`: `createRestorer({modelUrl, vocabUrl, metaUrl, ort}) -> {restore(text) -> Promise<string>}`.
Tests (node --test): windows identical to Python cases; `restoreWithProbs` with fake probs (1.0 at candidates) converts all candidates, keeps `ə`, keeps URLs' non-candidates, handles `""` and multi-line; 3000-char input keeps length.

### Task 4: Chrome extension
Files: `extension/manifest.json`, `background.js`, `offscreen.html`, `offscreen.js`, `content.js`, `popup.html`, `popup.js`, `options.html`, `options.js`, `styles.css`, `icons/*.png`, `extension/README.md`, `scripts/fetch_ort.py`, `scripts/pack_extension.py` (copies `web/shared/*` + `exported/<best>/` into `extension/model/` and ORT files into `extension/vendor/`).
Behaviour: shortcut / context menu → content script gets selection or whole focused field text → background → offscreen (ORT) or API mode (options: URL) → content script replaces via `execCommand('insertText')` (fallback: value setter + `input` event). Popup: textarea "yapışdır → düzəlt → kopyala", status of model, link to options.
Tests: `web/tests/extension.test.mjs` checks manifest permissions/CSP/commands and that every file referenced by manifest/html exists.

### Task 5: Website
Files: `web/index.html`, `web/style.css`, `web/app.js`, `web/README.md`.
Content (Azerbaijani): hero + live demo textarea (same core + ORT), "Necə işləyir" (3 steps), extension install guide, team/course footer, GitHub link. Model files loaded from `web/model/` (copied by `scripts/pack_extension.py --web`). Test: `web/tests/site.test.mjs` asserts referenced local files exist.

### Task 6: API + Docker
Files: `serve/app.py`, `serve/requirements.txt`, `Dockerfile`, `.dockerignore`, `tests/test_api.py` (importorskip fastapi, torch).
`MODEL_FAST`/`MODEL_ACCURATE` env vars → checkpoint paths; lazy load; CORS for the extension; limits per Global Constraints. Dockerfile: `python:3.11-slim`, CPU torch wheel, `uvicorn serve.app:app --host 0.0.0.0 --port 8000`.
