import { createRestorer } from "./shared/xettat-ort.js";
import { diffSegments } from "./app-diff.js";

const $ = (id) => document.getElementById(id);
const input = $("input");
const output = $("output");
const button = $("restore");
const status = $("status");
let restorer = null;

function setStatus(text, kind = "") {
  status.textContent = text;
  status.dataset.kind = kind;
}

function render(src, out) {
  output.replaceChildren(
    ...diffSegments(src, out).map((s) => {
      if (!s.changed) return document.createTextNode(s.text);
      const m = document.createElement("mark");
      m.textContent = s.text;
      return m;
    }),
  );
  const n = diffSegments(src, out).filter((s) => s.changed).reduce((k, s) => k + Array.from(s.text).length, 0);
  setStatus(n ? `${n} hərf bərpa olundu` : "Dəyişiklik lazım olmadı", "ok");
}

async function load() {
  setStatus("Model yüklənir…");
  try {
    const ort = await import("./vendor/ort.wasm.min.mjs");
    ort.env.wasm.wasmPaths = new URL("vendor/", location.href).href;
    restorer = await createRestorer({ ort, baseUrl: "model/" });
    button.disabled = false;
    setStatus("Model hazırdır. Mətni yazın və «Düzəlt» basın.", "ok");
  } catch (e) {
    setStatus("Demo modeli bu saytda hələ yerləşdirilməyib. Extension-u quraşdırıb sınaya bilərsiniz.", "err");
    console.warn(e);
  }
}

button.addEventListener("click", async () => {
  if (!restorer || !input.value.trim()) return;
  button.disabled = true;
  const t0 = performance.now();
  const out = await restorer.restore(input.value);
  render(input.value, out);
  status.textContent += ` (${Math.round(performance.now() - t0)} ms)`;
  button.disabled = false;
});

input.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) button.click();
});

document.querySelectorAll("[data-example]").forEach((b) =>
  b.addEventListener("click", () => {
    input.value = b.dataset.example;
    button.click();
  }),
);

$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText(output.textContent);
  $("copy").textContent = "Kopyalandı";
  setTimeout(() => ($("copy").textContent = "Kopyala"), 1500);
});

load();
