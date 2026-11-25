// Injected on demand (shortcut / context menu). Restores the selection or the whole focused field.
(() => {
  const TOAST_ID = "xettat-toast";

  function toast(msg, isError = false) {
    document.getElementById(TOAST_ID)?.remove();
    const el = document.createElement("div");
    el.id = TOAST_ID;
    el.textContent = msg;
    Object.assign(el.style, {
      position: "fixed", zIndex: 2147483647, right: "16px", bottom: "16px", maxWidth: "360px",
      padding: "10px 14px", borderRadius: "10px", font: "14px/1.4 system-ui, sans-serif",
      color: "#fff", background: isError ? "#b42318" : "#0f5f5c", boxShadow: "0 6px 24px rgba(0,0,0,.25)",
    });
    document.documentElement.appendChild(el);
    setTimeout(() => el.remove(), isError ? 6000 : 2200);
  }

  function deepActiveElement() {
    let el = document.activeElement;
    while (el?.shadowRoot?.activeElement) el = el.shadowRoot.activeElement;
    return el;
  }

  function isTextInput(el) {
    if (!el) return false;
    if (el.tagName === "TEXTAREA") return true;
    return el.tagName === "INPUT" && /^(text|search|url|email|)$/i.test(el.type || "");
  }

  function editableRoot(el) {
    while (el && el !== document.body) {
      if (el.isContentEditable && !el.parentElement?.isContentEditable) return el;
      el = el.parentElement;
    }
    return null;
  }

  function collect() {
    const el = deepActiveElement();
    if (isTextInput(el)) {
      let start = el.selectionStart ?? 0;
      let end = el.selectionEnd ?? 0;
      if (start === end) { start = 0; end = el.value.length; }
      return { kind: "input", el, start, end, text: el.value.slice(start, end) };
    }
    const root = editableRoot(el) || editableRoot(window.getSelection()?.anchorNode?.parentElement);
    if (root) {
      const sel = window.getSelection();
      if (!sel.rangeCount || sel.isCollapsed || !root.contains(sel.anchorNode)) {
        const range = document.createRange();
        range.selectNodeContents(root);
        sel.removeAllRanges();
        sel.addRange(range);
      }
      return { kind: "editable", el: root, text: sel.toString() };
    }
    const sel = window.getSelection()?.toString();
    if (sel) return { kind: "readonly", text: sel };
    return null;
  }

  function replace(target, newText) {
    if (target.kind === "input") {
      const { el, start, end } = target;
      el.focus();
      el.setSelectionRange(start, end);
      // execCommand keeps the browser's Undo stack and lets frameworks (React, etc.) see the change
      if (!document.execCommand("insertText", false, newText)) {
        el.setRangeText(newText, start, end, "end");
        el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: newText }));
      }
      return;
    }
    target.el.focus();
    if (!document.execCommand("insertText", false, newText)) {
      const sel = window.getSelection();
      const range = sel.getRangeAt(0);
      range.deleteContents();
      range.insertNode(document.createTextNode(newText));
      target.el.dispatchEvent(new InputEvent("input", { bubbles: true, inputType: "insertText", data: newText }));
    }
  }

  async function run() {
    const target = collect();
    if (!target || !target.text.trim()) {
      toast("Əvvəlcə mətn sahəsinə klikləyin və ya mətni seçin.", true);
      return;
    }
    const res = await chrome.runtime.sendMessage({ type: "restore", text: target.text });
    if (!res?.ok) {
      toast(res?.error ?? "Xəttat xətası", true);
      return;
    }
    if (res.text === target.text) {
      toast("Dəyişiklik lazım olmadı ✓");
      return;
    }
    if (target.kind === "readonly") {
      await navigator.clipboard.writeText(res.text).catch(() => {});
      toast("Bu mətn redaktə olunmur — düzəldilmiş variant kopyalandı.");
      return;
    }
    replace(target, res.text);
    toast("Xəttat: mətn düzəldildi ✓  (geri qaytarmaq: Ctrl+Z)");
  }

  run().catch((e) => toast(String(e?.message ?? e), true));
})();
