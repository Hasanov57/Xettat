// Xəttat inference core — dependency-free, shared by the Chrome extension and the website.
// Mirrors src/xettat/infer.py + src/xettat/text.py:
//  * v1 models (1 logit/char): convert where P(convert) > threshold, only on e/i/o/u/c/s/g candidates
//  * v2 models (3 logits/char: keep/convert/delete): masked argmax — sh/ch/gh and a→ə
//  * URLs, e-mails, @mentions and #hashtags are never changed
//  * windows overlap by 64; each char takes the decision from the window where it is farthest from an edge

export const TO_AZ = Object.freeze({
  e: "ə", i: "ı", o: "ö", u: "ü", c: "ç", s: "ş", g: "ğ",
  E: "Ə", I: "İ", O: "Ö", U: "Ü", C: "Ç", S: "Ş", G: "Ğ",
});
export const TO_AZ_EXT = Object.freeze({ ...TO_AZ, a: "ə", A: "Ə" });
export const CANDIDATES = new Set(Object.keys(TO_AZ));
export const KEEP = 0, CONVERT = 1, DELETE = 2;
const DIGRAPH_HEADS = new Set("sScCgG");

// Same patterns as text.py (_PROTECTED); \w there is Unicode-aware, hence \p{L}\p{N}_ here.
const W = "[\\p{L}\\p{N}_]";
const PROTECTED = new RegExp(
  `(?:https?://|www\\.)\\S+|[\\p{L}\\p{N}_.+-]+@${W}+(?:\\.${W}+)+|(?<!${W})[@#]${W}+`, "gu");

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

const LETTERS = /\p{L}+/gu; // whole words for the English guard (mirrors text.py _WORD)

/**
 * @param {string[]} chars code points of one line
 * @param {Set<string>} [english] English-only words (lower case) that must never change
 */
export function protectedMask(chars, english = null) {
  const text = chars.join("");
  const mask = new Array(chars.length).fill(false);
  // map UTF-16 offsets (regex) to code-point indices
  const cpIndex = [];
  chars.forEach((ch, i) => { for (let k = 0; k < ch.length; k++) cpIndex.push(i); });
  cpIndex.push(chars.length);
  for (const m of text.matchAll(PROTECTED)) {
    for (let u = m.index; u < m.index + m[0].length; u++) mask[cpIndex[u]] = true;
  }
  if (english?.size) {
    for (const m of text.matchAll(LETTERS)) {
      if (!english.has(m[0].toLowerCase())) continue;
      for (let u = m.index; u < m.index + m[0].length; u++) mask[cpIndex[u]] = true;
    }
  }
  return mask;
}

export function allowedActions(chars, i) {
  const c = chars[i];
  if (c in TO_AZ_EXT) return new Set([KEEP, CONVERT]);
  if (c === "h" || c === "H") return i > 0 && DIGRAPH_HEADS.has(chars[i - 1]) ? new Set([KEEP, DELETE]) : new Set([KEEP]);
  return new Set();
}

export function applyActions(chars, actions, protectedArr = null) {
  let out = "";
  chars.forEach((c, i) => {
    if (protectedArr?.[i]) out += c;
    else if (actions[i] === DELETE && (c === "h" || c === "H") && i > 0 && DIGRAPH_HEADS.has(chars[i - 1])) { /* dropped */ }
    else if (actions[i] === CONVERT && c in TO_AZ_EXT) out += TO_AZ_EXT[c];
    else out += c;
  });
  return out;
}

function actionsV1(chars, offset, probs, L, threshold) {
  const a = new Array(L);
  for (let i = 0; i < L; i++) a[i] = CANDIDATES.has(chars[offset + i]) && probs[i] > threshold ? CONVERT : KEEP;
  return a;
}

function actionsV2(chars, offset, logits, L, aBias) {
  const a = new Array(L);
  for (let i = 0; i < L; i++) {
    const allowed = allowedActions(chars, offset + i);
    if (allowed.size < 2) { a[i] = KEEP; continue; }
    let best = KEEP, bestV = -Infinity;
    for (const k of allowed) {
      let v = logits[i * 3 + k];
      if (k === CONVERT && (chars[offset + i] === "a" || chars[offset + i] === "A")) v += aBias;
      if (v > bestV) { bestV = v; best = k; }
    }
    a[i] = best;
  }
  return a;
}

/**
 * @param {string} text
 * @param {(chunks: string[]) => Promise<Float32Array[]>} scoresFn  per chunk: P(convert) per char (numClasses=1,
 *        probabilities) or raw logits flattened [L*3] (numClasses=3)
 */
export async function restoreWithLogits(text, scoresFn, maxLen,
  { numClasses = 1, threshold = 0.5, overlap = 64, aBias = 0, english = null } = {}) {
  const lines = text.normalize("NFC").split("\n");
  const lineChars = lines.map((l) => Array.from(l));
  const pieces = []; // [lineIdx, start, end]
  lineChars.forEach((chars, li) => {
    if (!chars.join("").trim()) return;
    for (const [s, e] of windows(chars.length, maxLen, Math.min(overlap, Math.floor(maxLen / 2)))) pieces.push([li, s, e]);
  });
  if (!pieces.length) return lines.join("\n");
  const scores = await scoresFn(pieces.map(([li, s, e]) => lineChars[li].slice(s, e).join("")));
  const actions = lineChars.map((c) => new Array(c.length).fill(KEEP));
  const margin = lineChars.map((c) => new Array(c.length).fill(-1));
  pieces.forEach(([li, s, e], k) => {
    const L = e - s;
    const acts = numClasses === 3
      ? actionsV2(lineChars[li], s, scores[k], L, aBias)
      : actionsV1(lineChars[li], s, scores[k], L, threshold);
    for (let i = 0; i < L; i++) {
      const m = Math.min(i, L - 1 - i);
      if (m > margin[li][s + i]) { margin[li][s + i] = m; actions[li][s + i] = acts[i]; }
    }
  });
  return lineChars.map((chars, li) => applyActions(chars, actions[li], protectedMask(chars, english))).join("\n");
}

/** v1 convenience wrapper: probsFn returns P(convert) per character. */
export function restoreWithProbs(text, probsFn, maxLen, opts = {}) {
  return restoreWithLogits(text, probsFn, maxLen, { ...opts, numClasses: 1 });
}
