import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import { protectedMask, allowedActions, applyActions, restoreWithLogits } from "../shared/xettat-core.js";

const fx = JSON.parse(readFileSync(new URL("./fixtures/parity.json", import.meta.url), "utf8"));

test("protected spans identical to Python", () => {
  for (const { text, mask } of fx.protected) {
    assert.equal(protectedMask(Array.from(text)).map((b) => (b ? "1" : "0")).join(""), mask, text);
  }
});

test("allowed actions identical to Python", () => {
  for (const { src, i, allowed } of fx.allowed) {
    assert.deepEqual([...allowedActions(Array.from(src), i)].sort(), allowed, `${src}@${i}`);
  }
});

test("applyActions identical to Python", () => {
  for (const { src, labels, out } of fx.apply) assert.equal(applyActions(Array.from(src), labels), out, src);
});

test("v2 logits: masked argmax restores digraphs and a", async () => {
  // fake model: prefer CONVERT for candidates, DELETE for every h (masking decides what is legal)
  const logitsFn = async (chunks) =>
    chunks.map((c) => {
      const chars = Array.from(c);
      const out = new Float32Array(chars.length * 3);
      chars.forEach((ch, i) => { out[i * 3 + (ch === "h" || ch === "H" ? 2 : 1)] = 5; });
      return out;
    });
  assert.equal(await restoreWithLogits("sheher eh", logitsFn, 64, { numClasses: 3 }), "şəhər əh");
  assert.equal(await restoreWithLogits("adad @seher", logitsFn, 64, { numClasses: 3 }), "ədəd @seher");
});
