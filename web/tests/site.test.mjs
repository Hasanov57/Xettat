import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const WEB = new URL("../", import.meta.url).pathname;
const html = readFileSync(join(WEB, "index.html"), "utf8");

test("local assets referenced by index.html exist", () => {
  for (const [, ref] of html.matchAll(/(?:src|href)="([^"#:]+)"/g)) {
    if (ref.startsWith("model/") || ref.startsWith("vendor/")) continue; // built by pack_extension.py
    assert.ok(existsSync(join(WEB, ref)), ref);
  }
});

test("page is Azerbaijani and has the demo controls", () => {
  assert.match(html, /<html lang="az">/);
  for (const id of ["input", "output", "restore", "status"]) assert.match(html, new RegExp(`id="${id}"`));
});

test("diff marks only changed characters", async () => {
  const { diffSegments } = await import("../app-diff.js");
  assert.deepEqual(diffSegments("seher", "səhər"), [
    { text: "s", changed: false }, { text: "ə", changed: true }, { text: "h", changed: false },
    { text: "ə", changed: true }, { text: "r", changed: false },
  ].reduce((acc, s) => {
    const last = acc.at(-1);
    if (last && last.changed === s.changed) last.text += s.text; else acc.push({ ...s });
    return acc;
  }, []));
  assert.deepEqual(diffSegments("", ""), []);
});
