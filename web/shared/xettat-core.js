// Xəttat inference core — dependency-free, shared by the Chrome extension and the website.
// Mirrors src/xettat/infer.py: only candidate letters may change, windows overlap by 64 and
// each character takes the probability from the window where it is farthest from an edge.

export const TO_AZ = Object.freeze({
  e: "ə", i: "ı", o: "ö", u: "ü", c: "ç", s: "ş", g: "ğ",
  E: "Ə", I: "İ", O: "Ö", U: "Ü", C: "Ç", S: "Ş", G: "Ğ",
});
export const CANDIDATES = new Set(Object.keys(TO_AZ));

export function makeVocab(chars) {
  const index = new Map(chars.map((ch, i) => [ch, i + 2])); // 0 = pad, 1 = unk
  return {
    size: chars.length + 2,
    encode(str) {
      return Array.from(str, (ch) => index.get(ch) ?? 1);
    },
  };
}

export function windows(n, maxLen, overlap) {
  if (n <= maxLen) return [[0, n]];
  const step = maxLen - overlap;
  const out = [];
  for (let s = 0; ; s += step) {
    const e = Math.min(s + maxLen, n);
    out.push([e - maxLen, e]);
    if (e === n) return out;
  }
}

/**
 * @param {string} text
 * @param {(chunks: string[]) => Promise<Float32Array[]>} probsFn  P(convert) per char of each chunk
 * @param {number} maxLen
 */
export async function restoreWithProbs(text, probsFn, maxLen, { threshold = 0.5, overlap = 64 } = {}) {
  const lines = text.normalize("NFC").split("\n");
  const pieces = []; // [lineIdx, start, end]
  lines.forEach((line, li) => {
    if (!line.trim()) return;
    const chars = Array.from(line);
    for (const [s, e] of windows(chars.length, maxLen, Math.min(overlap, Math.floor(maxLen / 2)))) {
      pieces.push([li, s, e]);
    }
  });
  if (!pieces.length) return lines.join("\n");
  const lineChars = lines.map((l) => Array.from(l));
  const probs = await probsFn(pieces.map(([li, s, e]) => lineChars[li].slice(s, e).join("")));
  const best = lineChars.map((c) => new Float32Array(c.length));
  const margin = lineChars.map((c) => new Float32Array(c.length).fill(-1));
  pieces.forEach(([li, s, e], k) => {
    const L = e - s;
    for (let i = 0; i < L; i++) {
      const m = Math.min(i, L - 1 - i);
      if (m > margin[li][s + i]) {
        margin[li][s + i] = m;
        best[li][s + i] = probs[k][i];
      }
    }
  });
  return lineChars
    .map((chars, li) => chars.map((ch, i) => (CANDIDATES.has(ch) && best[li][i] > threshold ? TO_AZ[ch] : ch)).join(""))
    .join("\n");
}
