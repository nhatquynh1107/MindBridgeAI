const $ = (id) => document.getElementById(id);

const chatBox = $("chatBox");
const msgInput = $("msgInput");
const sendBtn = $("sendBtn");
const modeSelect = $("modeSelect");
const streamToggle = $("streamToggle");
const ragToggle = $("ragToggle");
const sessionIdEl = $("sessionId");
const statusEl = $("status");
const newChatBtn = $("newChatBtn");
const clearBtn = $("clearBtn");

const fileInput = $("fileInput");
const uploadBtn = $("uploadBtn");
const ragStatusEl = $("ragStatus");

let sessionId = localStorage.getItem("session_id") || "";

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "user" : "bot"}`;
  div.textContent = text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

function setStatus(s) { statusEl.textContent = s || ""; }

async function apiNewSession() {
  const res = await fetch("/api/session/new", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("session_id", sessionId);
  sessionIdEl.textContent = sessionId;
  chatBox.innerHTML = "";
  await refreshRagStatus();
}

async function apiClear() {
  const res = await fetch("/api/session/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  });
  if (!res.ok) throw new Error(await res.text());
  chatBox.innerHTML = "";
  await refreshRagStatus();
}

async function refreshRagStatus() {
  if (!sessionId) return;
  try {
    const res = await fetch(`/api/rag/status?session_id=${encodeURIComponent(sessionId)}`);
    if (!res.ok) { ragStatusEl.textContent = ""; return; }
    const data = await res.json();
    ragStatusEl.textContent = `RAG chunks: ${data.total_chunks}`;
  } catch {
    ragStatusEl.textContent = "";
  }
}

async function loadModes() {
  const res = await fetch("/api/modes");
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  const modes = data.modes || [];

  modeSelect.innerHTML = "";
  for (const m of modes) {
    const opt = document.createElement("option");
    opt.value = m;
    opt.textContent = m;
    modeSelect.appendChild(opt);
  }

  const saved = localStorage.getItem("mode") || "";
  if (saved && modes.includes(saved)) {
    modeSelect.value = saved;
  } else if (modes.length) {
    modeSelect.value = modes[0];
    localStorage.setItem("mode", modes[0]);
  }
}

modeSelect.addEventListener("change", () => {
  localStorage.setItem("mode", modeSelect.value);
});

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  addMsg("user", text);
  msgInput.value = "";
  setStatus("Thinking...");

  const mode = modeSelect.value;
  const useRag = ragToggle.checked;
  const stream = streamToggle.checked;

  if (!stream) {
    try {
      const res = await fetch("/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ session_id: sessionId, message: text, mode, use_rag: useRag })
      });
      const raw = await res.text();
      if (!res.ok) throw new Error(raw);
      const data = JSON.parse(raw);
      addMsg("bot", data.reply || "");
      setStatus("");
    } catch (e) {
      addMsg("bot", `❌ Error: ${e.message}`);
      setStatus("");
    }
    return;
  }

  const botDiv = addMsg("bot", "");
  try {
    const res = await fetch("/api/chat/stream", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ session_id: sessionId, message: text, mode, use_rag: useRag })
    });
    if (!res.ok || !res.body) {
      const raw = await res.text();
      throw new Error(raw || `HTTP ${res.status}`);
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder("utf-8");
    let acc = "";

    while (true) {
      const { value, done } = await reader.read();
      if (done) break;
      const chunk = decoder.decode(value, { stream: true });
      acc += chunk;
      botDiv.textContent = acc;
      chatBox.scrollTop = chatBox.scrollHeight;
    }
    setStatus("");
  } catch (e) {
    botDiv.textContent += `\n\n❌ Error: ${e.message}`;
    setStatus("");
  }
}

sendBtn.addEventListener("click", sendMessage);
msgInput.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    sendMessage();
  }
});

newChatBtn.addEventListener("click", () =>
  apiNewSession().catch(err => addMsg("bot", `❌ Error: ${err.message}`))
);
clearBtn.addEventListener("click", () =>
  apiClear().catch(err => addMsg("bot", `❌ Error: ${err.message}`))
);

uploadBtn.addEventListener("click", async () => {
  if (!fileInput.files || fileInput.files.length === 0) return;
  const f = fileInput.files[0];
  setStatus("Uploading & indexing...");

  try {
    const form = new FormData();
    form.append("file", f);

    const res = await fetch(`/api/rag/upload?session_id=${encodeURIComponent(sessionId)}`, {
      method: "POST",
      body: form
    });
    const raw = await res.text();
    if (!res.ok) throw new Error(raw);
    await refreshRagStatus();
    setStatus("Upload OK.");
    setTimeout(() => setStatus(""), 1200);
  } catch (e) {
    setStatus("");
    addMsg("bot", `❌ Upload error: ${e.message}`);
  }
});

(async function init() {
  try {
    await loadModes();
    if (!sessionId) await apiNewSession();
    sessionIdEl.textContent = sessionId;
    await refreshRagStatus();
  } catch (e) {
    addMsg("bot", `❌ Init error: ${e.message}`);
  }
})();
