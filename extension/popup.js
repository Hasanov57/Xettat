const $ = (id) => document.getElementById(id);

async function refreshStatus() {
  const s = await chrome.runtime.sendMessage({ type: "status" }).catch((e) => ({ ok: false, detail: String(e) }));
  $("dot").className = "dot " + (s?.ok ? "ok" : "err");
  $("status").textContent = s?.detail ?? "Naməlum vəziyyət";
}

$("fix").addEventListener("click", async () => {
  const text = $("text").value;
  if (!text.trim()) return;
  $("fix").disabled = true;
  $("fix").textContent = "Düzəldilir…";
  const res = await chrome.runtime.sendMessage({ type: "restore", text });
  $("fix").disabled = false;
  $("fix").textContent = "Düzəlt";
  if (res?.ok) $("text").value = res.text;
  else { $("dot").className = "dot err"; $("status").textContent = res?.error ?? "Xəta"; }
});

$("copy").addEventListener("click", async () => {
  await navigator.clipboard.writeText($("text").value);
  $("copy").textContent = "Kopyalandı ✓";
  setTimeout(() => ($("copy").textContent = "Kopyala"), 1500);
});

$("text").addEventListener("keydown", (e) => {
  if (e.key === "Enter" && (e.ctrlKey || e.metaKey)) $("fix").click();
});

refreshStatus();
