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

let sessionId = localStorage.getItem("session_id") || "";

function addMsg(role, text) {
  const div = document.createElement("div");
  div.className = `msg ${role === "user" ? "user" : "bot"}`;
  div.innerHTML = window.marked?.parse ? marked.parse(text) : text;
  chatBox.appendChild(div);
  chatBox.scrollTop = chatBox.scrollHeight;
  return div;
}

function setStatus(s) { statusEl.textContent = s || ""; }

async function ensureSession() {
  sessionId = localStorage.getItem("session_id") || "";
  if (!sessionId || sessionId.length < 6) {
    await apiNewSession();
  }
}

async function apiNewSession() {
  const res = await fetch("/api/session/new", { method: "POST" });
  if (!res.ok) throw new Error(await res.text());
  const data = await res.json();
  sessionId = data.session_id;
  localStorage.setItem("session_id", sessionId);
  sessionIdEl.textContent = sessionId;
  chatBox.innerHTML = "";
}

async function apiClear() {
  const res = await fetch("/api/session/clear", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId })
  });
  if (!res.ok) throw new Error(await res.text());
  chatBox.innerHTML = "";
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

modeSelect.addEventListener("change", async () => {
  localStorage.setItem("mode", modeSelect.value);

  try {
    setStatus("Switching mode... clearing chat");
    await ensureSession();
    await apiClear();
    msgInput.value = "";
    msgInput.focus();
  } catch (e) {
    addMsg("bot", `❌ Auto-clear on mode change failed: ${e.message}`);
  } finally {
    setStatus("");
  }
});

async function sendMessage() {
  const text = msgInput.value.trim();
  if (!text) return;

  await ensureSession();

  if (!sessionId || sessionId.length < 6) {
    try {
      await apiNewSession();
    } catch (e) {
      addMsg("bot", `❌ Cannot create session: ${e.message}`);
      return;
    }
  }

  addMsg("user", text);
  msgInput.value = "";
  setStatus("Thinking...");

  const mode = modeSelect.value;
  const useRag = false;
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
      botDiv.innerHTML = window.marked?.parse ? marked.parse(acc) : acc;
      chatBox.scrollTop = chatBox.scrollHeight;
    }
    setStatus("");
  } catch (e) {
    botDiv.innerHTML += `\n\n❌ Error: ${e.message}`;
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

(async function init() {
  try {
    sendBtn.disabled = true;
    await loadModes();
    await ensureSession();
    if (!sessionId || sessionId.length < 6) await apiNewSession();
    sessionIdEl.textContent = sessionId;
  } catch (e) {
    addMsg("bot", `❌ Init error: ${e.message}`);
  } finally {
    sendBtn.disabled = false;
  }
})();

