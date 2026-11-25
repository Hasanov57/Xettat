import test from "node:test";
import assert from "node:assert/strict";
import { readFileSync, existsSync } from "node:fs";
import { join } from "node:path";

const EXT = new URL("../../extension/", import.meta.url).pathname;
const manifest = JSON.parse(readFileSync(join(EXT, "manifest.json"), "utf8"));

test("manifest v3 with minimal permissions", () => {
  assert.equal(manifest.manifest_version, 3);
  assert.deepEqual([...manifest.permissions].sort(), ["activeTab", "contextMenus", "offscreen", "scripting", "storage"]);
  assert.ok(!manifest.host_permissions, "no always-on host permissions");
  assert.match(manifest.content_security_policy.extension_pages, /'wasm-unsafe-eval'/);
  assert.equal(manifest.commands["restore-text"].suggested_key.default, "Alt+Shift+X");
});

test("every file referenced by manifest exists", () => {
  const refs = [manifest.background.service_worker, manifest.action.default_popup, manifest.options_page,
    ...Object.values(manifest.icons), ...Object.values(manifest.action.default_icon)];
  for (const r of refs) assert.ok(existsSync(join(EXT, r)), r);
});

test("every local script/link in extension html exists", () => {
  for (const page of ["popup.html", "options.html", "offscreen.html"]) {
    const html = readFileSync(join(EXT, page), "utf8");
    for (const [, ref] of html.matchAll(/(?:src|href)="([^"#:]+)"/g)) {
      assert.ok(existsSync(join(EXT, ref)), `${page} -> ${ref}`);
    }
  }
});

test("shared core copy is in sync with web/shared", () => {
  for (const f of ["xettat-core.js", "xettat-ort.js"]) {
    const a = readFileSync(join(EXT, "shared", f), "utf8");
    const b = readFileSync(new URL(`../shared/${f}`, import.meta.url), "utf8");
    assert.equal(a, b, `${f} out of sync: run python scripts/pack_extension.py`);
  }
});

test("no inline scripts (MV3 CSP)", () => {
  for (const page of ["popup.html", "options.html", "offscreen.html"]) {
    const html = readFileSync(join(EXT, page), "utf8");
    assert.doesNotMatch(html, /<script(?![^>]*src=)[^>]*>/, page);
    assert.doesNotMatch(html, /\son[a-z]+="/, page);
  }
});
