// Xəttat service worker: routes restore requests to the local model (offscreen doc) or the API.
const MENU_ID = "xettat-restore";

chrome.runtime.onInstalled.addListener(() => {
  chrome.contextMenus.create({ id: MENU_ID, title: "Xəttat ilə düzəlt", contexts: ["editable", "selection"] });
});

async function getSettings() {
  const { mode = "local", apiUrl = "http://localhost:8000" } = await chrome.storage.sync.get(["mode", "apiUrl"]);
  return { mode, apiUrl: apiUrl.replace(/\/+$/, "") };
}

let creating = null;
async function ensureOffscreen() {
  if (await chrome.offscreen.hasDocument()) return;
  creating ??= chrome.offscreen.createDocument({
    url: "offscreen.html",
    reasons: ["WORKERS"],
    justification: "Xəttat modelini (WebAssembly) lokal işlətmək üçün",
  });
  try { await creating; } finally { creating = null; }
}

async function restore(text) {
  const { mode, apiUrl } = await getSettings();
  if (mode === "api") {
    const r = await fetch(`${apiUrl}/restore`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text, model: "accurate" }),
    });
    if (!r.ok) throw new Error(`Server xətası: ${r.status}`);
    return (await r.json()).text;
  }
  await ensureOffscreen();
  const res = await chrome.runtime.sendMessage({ target: "offscreen", type: "restore", text });
  if (!res?.ok) throw new Error(res?.error ?? "Lokal model cavab vermədi");
  return res.text;
}

async function status() {
  const { mode, apiUrl } = await getSettings();
  if (mode === "api") {
    try {
      const r = await fetch(`${apiUrl}/health`);
      return { ok: r.ok, mode, detail: r.ok ? `Server: ${apiUrl}` : `Server cavab vermir (${r.status})` };
    } catch {
      return { ok: false, mode, detail: `Serverə qoşulmaq olmadı: ${apiUrl}` };
    }
  }
  await ensureOffscreen();
  const res = await chrome.runtime.sendMessage({ target: "offscreen", type: "status" });
  return { mode, ...(res ?? { ok: false, detail: "Lokal model cavab vermədi" }) };
}

chrome.runtime.onMessage.addListener((msg, _sender, sendResponse) => {
  if (msg?.target === "offscreen") return false; // not for us
  if (msg?.type === "restore") {
    restore(msg.text).then((text) => sendResponse({ ok: true, text }), (e) => sendResponse({ ok: false, error: String(e.message ?? e) }));
    return true;
  }
  if (msg?.type === "status") {
    status().then(sendResponse, (e) => sendResponse({ ok: false, detail: String(e.message ?? e) }));
    return true;
  }
  return false;
});

async function runInTab(tab, frameId) {
  if (!tab?.id) return;
  try {
    await chrome.scripting.executeScript({
      target: frameId != null ? { tabId: tab.id, frameIds: [frameId] } : { tabId: tab.id },
      files: ["content.js"],
    });
  } catch (e) {
    console.warn("Xəttat: bu səhifədə işləmək mümkün deyil", e); // chrome:// pages, Web Store, etc.
  }
}

chrome.commands.onCommand.addListener(async (command, tab) => {
  if (command !== "restore-text") return;
  tab ??= (await chrome.tabs.query({ active: true, currentWindow: true }))[0];
  runInTab(tab);
});

chrome.contextMenus.onClicked.addListener((info, tab) => {
  if (info.menuItemId === MENU_ID) runInTab(tab, info.frameId);
});
