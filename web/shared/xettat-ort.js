// Thin onnxruntime-web runner around xettat-core.js.
// Usage: const r = await createRestorer({ ort, baseUrl: "model/" }); await r.restore("seher");
import { makeVocab, restoreWithProbs } from "./xettat-core.js";

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

export async function createRestorer({ ort, baseUrl = "model/", modelFile = "model.int8.onnx", batchSize = 32 }) {
  const [vocabChars, meta] = await Promise.all([
    fetch(baseUrl + "vocab.json").then((r) => r.json()),
    fetch(baseUrl + "meta.json").then((r) => r.json()),
  ]);
  const vocab = makeVocab(vocabChars);
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
        const p = new Float32Array(L);
        for (let t = 0; t < L; t++) p[t] = sigmoid(logits[r * T + t]);
        out[i] = p;
      });
    }
    return out;
  }

  return {
    meta,
    restore: (text) => restoreWithProbs(text, probsFn, meta.max_len ?? 512, { threshold: meta.threshold ?? 0.5 }),
  };
}
