const $ = (id) => document.getElementById(id);

const { mode = "local", apiUrl = "http://localhost:8000" } = await chrome.storage.sync.get(["mode", "apiUrl"]);
document.querySelector(`input[name=mode][value=${mode}]`).checked = true;
$("apiUrl").value = apiUrl;

$("save").addEventListener("click", async () => {
  const newMode = document.querySelector("input[name=mode]:checked").value;
  const url = $("apiUrl").value.trim() || "http://localhost:8000";
  if (newMode === "api") {
    const origin = new URL(url).origin + "/*";
    const granted = await chrome.permissions.request({ origins: [origin] });
    if (!granted) { $("saved").textContent = "Server üçün icazə verilmədi."; return; }
  }
  await chrome.storage.sync.set({ mode: newMode, apiUrl: url });
  $("saved").textContent = "Yadda saxlanıldı ✓";
  setTimeout(() => ($("saved").textContent = ""), 2000);
});
