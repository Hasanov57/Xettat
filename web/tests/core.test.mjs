import test from "node:test";
import assert from "node:assert/strict";
import { CANDIDATES, TO_AZ, makeVocab, windows, restoreWithProbs } from "../shared/xettat-core.js";

const allOnes = async (chunks) => chunks.map((c) => new Float32Array(c.length).fill(1));
const allZeros = async (chunks) => chunks.map((c) => new Float32Array(c.length).fill(0));

test("map has 14 candidates", () => {
  assert.equal(CANDIDATES.size, 14);
  assert.equal(TO_AZ.e, "ə");
  assert.equal(TO_AZ.I, "İ");
});

test("windows match python", () => {
  assert.deepEqual(windows(10, 64, 16), [[0, 10]]);
  const w = windows(3000, 512, 64);
  assert.equal(w[0][0], 0);
  assert.equal(w.at(-1)[1], 3000);
  assert.ok(w.every(([s, e]) => e - s === 512));
  assert.deepEqual(w.slice(0, 2), [[0, 512], [448, 960]]);
});

test("vocab encodes unknown as 1 and never 0", () => {
  const v = makeVocab(["a", "b"]);
  assert.deepEqual(Array.from(v.encode("ab😀")), [2, 3, 1]);
});

test("restore converts candidates only and keeps typed letters", async () => {
  assert.equal(await restoreWithProbs("seher ə 123", allOnes, 64), "şəhər ə 123");
  assert.equal(await restoreWithProbs("seher", allZeros, 64), "seher");
});

test("empty, whitespace, multiline", async () => {
  assert.equal(await restoreWithProbs("", allOnes, 64), "");
  assert.equal(await restoreWithProbs("  \n ", allOnes, 64), "  \n ");
  assert.equal(await restoreWithProbs("ab\n\nce", allOnes, 64), "ab\n\nçə");
});

test("long input keeps length", async () => {
  const s = "seher ".repeat(500);
  const out = await restoreWithProbs(s, allOnes, 128);
  assert.equal(out.length, s.length);
  assert.equal(out, "şəhər ".repeat(500));
});

test("window centre preference", async () => {
  // probs = 1 only where the char is far from its window edge -> every position covered by some centre
  const s = "e".repeat(300);
  const centreOnly = async (chunks) =>
    chunks.map((c) => Float32Array.from(c, (_, i) => (Math.min(i, c.length - 1 - i) >= 10 || c.length < 100 ? 1 : 0)));
  const out = await restoreWithProbs(s, centreOnly, 128);
  // first/last 10 chars of the whole text have no centre window -> stay "e"
  assert.equal(out.slice(20, 280), "ə".repeat(260));
});
