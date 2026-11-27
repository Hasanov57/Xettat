// Thin onnxruntime-web runner around xettat-core.js.
// Usage: const r = await createRestorer({ ort, baseUrl: "model/" }); await r.restore("seher");
import { makeVocab, restoreWithLogits } from "./xettat-core.js";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export async function createRestorer({ ort, baseUrl = "model/", modelFile = "model.int8.onnx", batchSize = 32 }) {
  const [vocabChars, meta] = await Promise.all([
    fetch(baseUrl + "vocab.json").then((r) => r.json()),
    fetch(baseUrl + "meta.json").then((r) => r.json()),
  ]);
  const vocab = makeVocab(vocabChars);
  // English-only words that are never restored (curve, meeting, ...); optional file next to this module
  const english = await fetch(new URL("./english-guard.txt", import.meta.url))
    .then((r) => (r.ok ? r.text() : ""))
    .then((t) => new Set(t.split(/\s+/).filter(Boolean)))
    .catch(() => new Set());
  batchSize = meta.batch_size ?? batchSize; // BiLSTM exports run unpadded, one line per call
  const C = meta.num_classes ?? 1; // 1 = v1 (binary), 3 = v2 (keep/convert/delete)
  const session = await ort.InferenceSession.create(baseUrl + (meta.default_file ?? modelFile), {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });

  async function probsFn(chunks) {
    const out = new Array(chunks.length);
    const order = chunks.map((c, i) => i).sort((a, b) => chunks[a].length - chunks[b].length);
    for (let b = 0; b < order.length; b += batchSize) {
      const idx = order.slice(b, b + batchSize);
      const encoded = idx.map((i) => vocab.encode(chunks[i]));
      const T = Math.max(...encoded.map((e) => e.length));
      const ids = new BigInt64Array(idx.length * T);
      const mask = new Uint8Array(idx.length * T);
      encoded.forEach((e, r) => e.forEach((v, t) => { ids[r * T + t] = BigInt(v); mask[r * T + t] = 1; }));
      const res = await session.run({
        ids: new ort.Tensor("int64", ids, [idx.length, T]),
        mask: new ort.Tensor("bool", mask, [idx.length, T]),
      });
      const logits = res.logits.data;
      idx.forEach((i, r) => {
        const L = encoded[r].length;
        if (C === 1) {
          const p = new Float32Array(L);
          for (let t = 0; t < L; t++) p[t] = sigmoid(logits[r * T + t]);
          out[i] = p;
        } else {
          out[i] = logits.slice(r * T * C, r * T * C + L * C); // raw logits, masked argmax in the core
        }
      });
    }
    return out;
  }

  return {
    meta,
    restore: (text) => restoreWithLogits(text, probsFn, meta.max_len ?? 512,
      { numClasses: C, threshold: meta.threshold ?? 0.5, aBias: meta.a_bias ?? 0, english }),
  };
}
