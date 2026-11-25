// Hosts onnxruntime-web (WASM) for the extension. Loaded lazily by background.js.
import { createRestorer } from "./shared/xettat-ort.js";

let restorerPromise = null;

async function load() {
  const ort = await import("./vendor/ort.wasm.min.mjs");
  ort.env.wasm.wasmPaths = chrome.runtime.getURL("vendor/");
  ort.env.wasm.numThreads = 1; // no cross-origin isolation in extension pages
  return createRestorer({ ort, baseUrl: chrome.runtime.getURL("model/") });
}

function restorer() {
  restorerPromise ??= load().catch((e) => {
    restorerPromise = null;
    throw new Error(
      "Model faylları tapılmadı və ya yüklənmədi. extension/README.md-dəki 'Modeli əlavə etmək' addımlarına baxın. (" +
        (e?.message ?? e) + ")",
    );
  });
  return restorerPromise;
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target !== "offscreen") return false;
  if (msg.type === "restore") {
    restorer()
      .then((r) => r.restore(msg.text))
      .then((text) => sendResponse({ ok: true, text }), (e) => sendResponse({ ok: false, error: String(e.message ?? e) }));
    return true;
  }
  if (msg.type === "status") {
    restorer().then(
      (r) => sendResponse({ ok: true, detail: `Lokal model hazırdır (${r.meta.model_type ?? "model"})` }),
      (e) => sendResponse({ ok: false, detail: String(e.message ?? e) }),
    );
    return true;
  }
  return false;
});
