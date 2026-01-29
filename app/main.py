from __future__ import annotations

import os
import re
import uuid
from typing import Generator, List, Optional, Dict, Any

from fastapi import FastAPI, File, HTTPException, UploadFile, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles

import google.generativeai as genai

from .prompts import MODES, get_system_prompt
from .rag import chunk_by_words, cosine_top_k, read_pdf_file, read_text_file
from .schemas import ChatRequest, ChatResponse, ClearRequest, NewSessionResponse
from .store import RAGChunk, store

from pathlib import Path
from .local_llm import local_generate_reply

CRISIS_KEYWORDS = [
    "kill myself",
    "suicide",
    "self harm",
    "self-harm",
    "cut myself",
    "i want to die",
    "end my life",
]

CRISIS_RESPONSE = (
    "I’m really sorry you’re feeling this way. You deserve support and you don’t have to face this alone.\n\n"
    "If you might hurt yourself or feel in immediate danger, please contact your local emergency number right now.\n"
    "If you can, tell a trusted adult (parent/guardian/teacher/school counselor) immediately.\n"
    "If you tell me your country, I can suggest crisis hotline options."
)


def is_crisis(text: str) -> bool:
    t = (text or "").lower()
    return any(k in t for k in CRISIS_KEYWORDS)

from dotenv import load_dotenv
load_dotenv()

def is_demo() -> bool:
    v = (os.getenv("DEMO_MOCK") or "").strip().lower()
    return v in ("1", "true", "yes", "on")

def is_local_only() -> bool:
    return os.getenv("LOCAL_ONLY", "0").lower() in ("1", "true", "yes", "on")


KB_DIR = Path(__file__).resolve().parent / "knowledge"

def autoload_builtin_kb(session_id: str) -> None:
    # auto load app/knowledge/*.md into session RAG chunks (no embeddings needed)
    if os.getenv("AUTO_LOAD_KB", "1").lower() in ("0", "false", "no", "off"):
        return

    if store.get_rag_chunks(session_id):
        return

    if not KB_DIR.exists():
        return

    rag_chunks: List[RAGChunk] = []
    max_chunks_per_doc = int(os.getenv("KB_MAX_CHUNKS_PER_DOC", "40"))

    for f in sorted(KB_DIR.glob("*.md")):
        text = f.read_text(encoding="utf-8", errors="ignore")
        text = text.replace("\r\n", "\n").replace("\r", "\n")
        text = re.sub(r"\n{3,}", "\n\n", text).strip()
        chunks = chunk_by_words(text)[:max_chunks_per_doc]
        for i, ch in enumerate(chunks):
            rag_chunks.append(
                RAGChunk(
                    doc_name=f.name,
                    chunk_id=f"{f.name}#{i}",
                    text=ch,
                    embedding=[0.0],  # placeholder
                )
            )

    if rag_chunks:
        store.add_rag_chunks(session_id, rag_chunks)

def require_api_key() -> None:
    if is_demo() or is_local_only():
        return
    if not os.getenv("GEMINI_API_KEY"):
        raise HTTPException(
            status_code=500,
            detail="Missing GEMINI_API_KEY. Set environment variable GEMINI_API_KEY.",
        )


def configure_gemini():
    require_api_key()
    genai.configure(api_key=os.getenv("GEMINI_API_KEY"))


def get_gemini_model(system_instruction: str = None):
    configure_gemini()
    model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
    return genai.GenerativeModel(model_name, system_instruction=system_instruction)


app = FastAPI(title="TechSprint MVP AI Chatbot", version="0.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


@app.get("/")
def root():
    return FileResponse("static/index.html")

@app.get("/health")
def health():
    backend = "ollama" if is_local_only() else ("demo" if is_demo() else "gemini")
    return {
        "ok": True,
        "backend": backend,
        "demo": is_demo(),
        "local_only": is_local_only(),
        "has_gemini_key": bool(os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")),
    }

@app.get("/api/modes")
def modes():
    return {"modes": MODES}


@app.post("/api/session/new", response_model=NewSessionResponse)
def new_session():
    sid = uuid.uuid4().hex
    store.get_or_create(sid)
    autoload_builtin_kb(sid)  
    return NewSessionResponse(session_id=sid)

@app.post("/api/session/clear")
def clear_session(payload: ClearRequest):
    store.clear(payload.session_id)
    return {"ok": True, "session_id": payload.session_id}


def build_instructions(mode: str, rag_context: Optional[str]) -> str:
    base = get_system_prompt(mode)
    if rag_context:
        base += (
            "\n\nYou have USER-PROVIDED NOTES below. Use them when relevant. "
            "If the answer is not in notes, say so. "
            "When referencing notes, cite chunk ids like [doc#chunk].\n\n"
            f"NOTES:\n{rag_context}\n"
        )
    return base

def make_rag_context_local(session_id: str, query: str, k: int = 4) -> str:
    chunks = store.get_rag_chunks(session_id)
    if not chunks:
        return ""

    q = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    scored = []
    for c in chunks:
        t = set(re.findall(r"[a-zA-Z0-9_]+", (c.text or "").lower()))
        score = len(q & t)
        if score > 0:
            scored.append((c.chunk_id, score, c.text))

    scored.sort(key=lambda x: x[1], reverse=True)
    top = scored[:k]
    return "\n\n".join([f"[{cid}] {txt[:900]}" for cid, _s, txt in top])

def make_rag_context(session_id: str, query: str, k: int = 4) -> str:
    chunks = store.get_rag_chunks(session_id)
    if not chunks:
        return ""

    try:
        configure_gemini()
        result = genai.embed_content(
            model="models/text-embedding-004",
            content=query,
            task_type="retrieval_query"
        )
        qv = result['embedding']
    except Exception as e:
        print(f"Embedding error: {e}")
        return ""

    items = [(c.chunk_id, c.embedding, c.text) for c in chunks]
    top = cosine_top_k(qv, items, k=k)
    if not top:
        return ""

    lines: List[str] = []
    for chunk_id, _score, text in top:
        lines.append(f"[{chunk_id}] {text[:900]}")
    return "\n\n".join(lines)


def _overlap_score(query: str, text: str) -> int:
    q = set(re.findall(r"[a-zA-Z0-9_]+", query.lower()))
    t = set(re.findall(r"[a-zA-Z0-9_]+", text.lower()))
    return len(q & t)


def make_rag_context_demo(session_id: str, query: str, k: int = 4) -> str:
    chunks = store.get_rag_chunks(session_id)
    if not chunks:
        return ""

    scored = [(c.chunk_id, _overlap_score(query, c.text), c.text) for c in chunks]
    scored.sort(key=lambda x: x[1], reverse=True)
    top = [x for x in scored if x[1] > 0][:k]
    if not top:
        return ""

    lines = [f"[{cid}] {txt[:900]}" for cid, _s, txt in top]
    return "\n\n".join(lines)


def demo_health(message: str) -> str:
    low = (message or "").lower()

    if any(x in low for x in ["panic", "panic attack", "anxious", "anxiety", "overthinking", "stressed"]):
        return (
            "I’m here with you. Want to try a quick reset?\n\n"
            "1) Breathe in for 4, hold 2, out for 6 (repeat 5 times)\n"
            "2) Name 5 things you see, 4 you feel, 3 you hear\n\n"
            "What’s the biggest thing on your mind right now?"
        )

    if any(x in low for x in ["tired", "exhausted", "burnt out", "burnout"]):
        return (
            "That sounds exhausting. Let’s make this lighter.\n"
            "For the next 10 minutes: drink water, stretch, and do one tiny task you can finish.\n"
            "What’s one thing you need to get through today?"
        )

    if any(x in low for x in ["sad", "down", "depressed", "hopeless", "upset"]):
        return (
            "I’m really sorry you’re feeling this way. You’re not alone here.\n"
            "If you want, tell me what happened today or what’s been building up.\n"
            "What would feel like a small relief right now?"
        )

    return (
        "I hear you. I can support you with calming tools, gentle plans, or just listening.\n"
        "What’s been going on lately?"
    )


def _last_assistant_text(session_id: str) -> str:
    hist = store.get_history(session_id)
    for m in reversed(hist):
        if m.get("role") == "assistant":
            return m.get("content", "") or ""
    return ""


def demo_health_flow(session_id: str, message: str) -> str:
    last_bot = _last_assistant_text(session_id)
    m = (message or "").strip()
    low = m.lower()

    if "What's one thing you need to get through today?" in last_bot:
        if any(k in low for k in ["ddl", "deadline", "deadlines", "due", "homework", "assignment", "exam", "test", "quiz", "workload"]):
            return (
                "Okay — deadlines. Let’s make it lighter and concrete.\n\n"
                "Quick plan (10 minutes):\n"
                "1) Write the list of tasks (2 min)\n"
                "2) Pick the top 2 by urgency (1 min)\n"
                "3) Start the smallest first step on #1 (7 min)\n\n"
                "Tell me the nearest due time/date + the 2 tasks, and I’ll order them for you."
            )
        return (
            "Got it. Let’s keep it simple.\n"
            "For the next 10 minutes: water, stretch, then do ONE tiny step you can finish.\n\n"
            "What’s the nearest deadline (time/date)?"
        )

    if "What’s been going on lately?" in last_bot or "What's been going on lately?" in last_bot:
        if any(k in low for k in ["tired", "exhausted", "burnt out", "burned out"]):
            return (
                "That sounds exhausting.\n\n"
                "Quick reset (2 minutes):\n"
                "1) Drink water\n"
                "2) Stand up + stretch\n"
                "3) 10 slow breaths\n\n"
                "Is it mainly sleep, workload, or stress?"
            )
        if any(k in low for k in ["ddl", "deadline", "deadlines", "due", "homework", "assignment", "exam", "test", "quiz"]):
            return (
                "Got you — deadlines can feel heavy.\n\n"
                "What’s your nearest deadline (date/time), and what are the tasks? "
                "List them in one line each."
            )
        return (
            "Thanks for sharing.\n"
            "What do you want right now: calming, a practical plan, or just venting?"
        )

    if any(x in low for x in ["tired", "exhausted", "burnt out", "burned out"]):
        return (
            "That sounds exhausting. Let’s make this lighter.\n"
            "For the next 10 minutes: drink water, stretch, and do one tiny task you can finish.\n"
            "What’s one thing you need to get through today?"
        )

    return (
        "I hear you. I can support you with calming tools, gentle plans, or just listening.\n"
        "What’s been going on lately?"
    )


def demo_school_stress(message: str) -> str:
    return (
        "Let’s reduce the overwhelm with a simple plan.\n\n"
        "1) What’s your nearest deadline (date/time)?\n"
        "2) List the next 3 tiny tasks (10–20 min each)\n"
        "3) Do one 25-minute focus block, then 5-minute break\n\n"
        "What assignment/exam is stressing you most right now?"
    )


def demo_school_stress_flow(session_id: str, message: str) -> str:
    last_bot = _last_assistant_text(session_id)
    subj = (message or "").strip()
    low = subj.lower()

    if "What assignment/exam is stressing you most right now?" in last_bot:
        subject = subj if subj else "that"
        return (
            f"Okay — {subject}. Let’s start with a quick, practical plan.\n\n"
            "1) Pick ONE topic inside it (example: equations / derivatives / geometry).\n"
            "2) Do 3 quick wins:\n"
            "   - 5 min: list what you must submit\n"
            "   - 10 min: solve 1 easiest question\n"
            "   - 15 min: solve 1 medium question\n\n"
            "Tell me: what exact math topic is it (algebra / calculus / geometry) and when is it due?"
        )

    if any(x in low for x in ["algebra", "equation", "equations", "linear", "quadratic"]):
        return (
            "Algebra — got it.\n\n"
            "Fast start (25 minutes):\n"
            "1) Write formulas you need (5 min)\n"
            "2) Do 2 easy questions (10 min)\n"
            "3) Do 1 medium question (10 min)\n\n"
            "Paste ONE question you’re stuck on (or the topic name) and I’ll guide you step-by-step."
        )

    if any(x in low for x in ["calculus", "derivative", "derivatives", "integral", "integrals", "limit", "limits"]):
        return (
            "Calculus — got it.\n\n"
            "Fast start (25 minutes):\n"
            "1) Write the key rules (5 min)\n"
            "2) Do 2 basic derivatives/limits (10 min)\n"
            "3) Do 1 mixed question (10 min)\n\n"
            "Send one problem or tell me which part (limits / derivatives / integrals)."
        )

    if any(x in low for x in ["geometry", "triangle", "triangles", "circle", "circles"]):
        return (
            "Geometry — got it.\n\n"
            "Fast start (25 minutes):\n"
            "1) List theorems you might use (5 min)\n"
            "2) Solve 1 easiest question (10 min)\n"
            "3) Solve 1 medium question (10 min)\n\n"
            "Send a photo/text of one question and I’ll walk you through the steps."
        )

    return (
        "Let’s reduce the overwhelm with a simple plan.\n\n"
        "1) What’s your nearest deadline (date/time)?\n"
        "2) List the next 3 tiny tasks (10–20 min each)\n"
        "3) Do one 25-minute focus block, then 5-minute break\n\n"
        "What assignment/exam is stressing you most right now?"
    )


def demo_friends(message: str) -> str:
    return (
        "That sounds tough. I won’t judge you or them.\n"
        "Tell me what happened and what you want (apology, clarity, space, or to move on).\n\n"
        "If you want a simple script:\n"
        "“Hey, I felt ___ when ___. I’d like __. Can we talk?”\n\n"
        "What outcome are you hoping for?"
    )


def demo_friends_flow(session_id: str, message: str) -> str:
    last_bot = _last_assistant_text(session_id)
    m = (message or "").strip()

    if "What outcome are you hoping for?" in last_bot:
        want = m if m else "an apology / clarity"
        return (
            f"Got it — you want {want}. Here’s a simple repair plan:\n\n"
            "1) Own it (no excuses): say what you did.\n"
            "2) Apologize specifically: name the impact.\n"
            "3) Repair: ask what would help + stop the behavior.\n"
            "4) Give space if needed.\n\n"
            "Message you can send:\n"
            "\"Hey, I want to apologize. I shared things I shouldn’t have, and that was disrespectful. "
            "I’m sorry for the stress or hurt it caused. I’ve stopped and I won’t do it again. "
            "If you’re open, I’d like to make it right — what would help?\"\n\n"
            "Quick question: do you want to apologize by text, or in person?"
        )

    if "do you want to apologize by text, or in person?" in last_bot.lower():
        if "text" in m.lower():
            return (
                "Text is fine — keep it short and clear.\n\n"
                "Use this:\n"
                "\"Hey, I’m sorry. I gossiped about you and that wasn’t okay. "
                "I understand it could hurt your trust. I’ve stopped and won’t do it again. "
                "If you’re willing, I’d like to make it right.\"\n\n"
                "Then stop talking and let them respond."
            )
        if "person" in m.lower() or "in person" in m.lower():
            return (
                "In person works best for trust repair.\n\n"
                "Say:\n"
                "1) \"I owe you an apology. I gossiped about you.\"\n"
                "2) \"That was wrong and I understand it could hurt you.\"\n"
                "3) \"I’ve stopped and won’t do it again.\"\n"
                "4) \"What would help you feel okay?\"\n\n"
                "Then listen without defending yourself."
            )

    if "Tell me what happened" in last_bot or "Tell me what happened" in last_bot.lower():
        return (
            "Thanks for telling me. Before we choose the best move:\n"
            "1) How did they find out (someone told them / they overheard / you admitted it)?\n"
            "2) Are they angry, distant, or just ignoring you?\n\n"
            "Based on your answer, I’ll suggest the cleanest next message."
        )

    return (
        "That sounds tough. I won’t judge you or them.\n"
        "Tell me what happened and what you want (apology, clarity, space, or to move on).\n\n"
        "If you want a simple script:\n"
        "\"Hey, I felt __ when __. I’d like __. Can we talk?\"\n\n"
        "What outcome are you hoping for?"
    )


def demo_coding(message: str) -> str:
    low = (message or "").lower()
    if any(x in low for x in ["error", "bug", "exception", "traceback", "segfault", "compile", "uvicorn", "fastapi"]):
        return (
            "To debug fast, send:\n"
            "1) The full error log\n"
            "2) The relevant code snippet/file\n"
            "3) Expected vs actual behavior\n\n"
            "Also confirm: venv active (which python), deps installed, env vars set, and port not in use."
        )

    return (
        "Tell me what you want to build (inputs/outputs + constraints) and I’ll write the code.\n"
        "If you paste your current code + error, I can fix it quickly."
    )


def demo_generate_reply(session_id: str, mode: str, message: str) -> str:
    if is_crisis(message):
        return CRISIS_RESPONSE

    low = (message or "").lower()

    school_kw = [
        "homework","assignment","deadline","deadlines","ddl","due",
        "exam","test","quiz","study","class","school","math","project","projects",
        "workload","overwhelmed","too much work"
    ]
    friend_kw = [
        "friend","friends","bestie","bully","argument","fight","ignored","left out",
        "relationship","boyfriend","girlfriend","gossip","apology"
    ]
    coding_kw = [
        "code","bug","error","exception","traceback","compile","uvicorn","fastapi",
        "python","javascript","c++"
    ]

    last_bot = _last_assistant_text(session_id).lower()

    if "what assignment/exam is stressing you most right now" in last_bot:
        mode = "School Stress"
    if "do you want to apologize by text" in last_bot and low.strip() in ["text", "in person", "person"]:
        mode = "Friends"

    if any(k in low for k in coding_kw):
        mode = "Coding"
    elif any(k in low for k in friend_kw):
        mode = "Friends"
    elif any(k in low for k in school_kw):
        mode = "School Stress"

    if mode == "School Stress":
        return demo_school_stress_flow(session_id, message)
    if mode == "Friends":
        return demo_friends_flow(session_id, message)
    if mode == "Health":
        return demo_health_flow(session_id, message)
    if mode == "Coding":
        return demo_coding(message)
    return demo_health_flow(session_id, message)


def _to_gemini_history(history: List[Dict[str, str]]):
    gemini_hist = []
    for msg in history:
        role = "user" if msg["role"] == "user" else "model"
        content = msg.get("content", "")
        if content:
             gemini_hist.append({"role": role, "parts": [content]})
    return gemini_hist


@app.post("/api/chat", response_model=ChatResponse)
def chat(payload: ChatRequest):
    if not payload.session_id or len(payload.session_id) < 6:
        payload.session_id = uuid.uuid4().hex
        store.get_or_create(payload.session_id)
    
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Empty input.")
    
    if is_crisis(payload.message):
        store.get_or_create(payload.session_id)
        store.append_history(payload.session_id, "user", payload.message)
        store.append_history(payload.session_id, "assistant", CRISIS_RESPONSE)
        return ChatResponse(session_id=payload.session_id, reply=CRISIS_RESPONSE, mode=payload.mode)

    if is_demo():
        store.get_or_create(payload.session_id)

        rag_context = ""
        if payload.use_rag:
            rag_context = make_rag_context_demo(payload.session_id, payload.message)

        answer = demo_generate_reply(payload.session_id, payload.mode, payload.message)
        if rag_context:
            answer += "\n\n---\nTop related notes (demo):\n" + rag_context

        store.append_history(payload.session_id, "user", payload.message)
        store.append_history(payload.session_id, "assistant", answer)
        return ChatResponse(session_id=payload.session_id, reply=answer, mode=payload.mode)

    # ===== LOCAL ONLY (Ollama) =====
    if is_local_only():
        store.get_or_create(payload.session_id)
        autoload_builtin_kb(payload.session_id)
        history = store.get_history(payload.session_id)

        rag_context = ""
        if payload.use_rag:
            rag_context = make_rag_context_local(payload.session_id, payload.message)

        instructions = build_instructions(payload.mode, rag_context)

        hist_tuples = [(m["role"], m.get("content", "")) for m in history if m.get("content")]

        try:
            answer = local_generate_reply(payload.mode, instructions, hist_tuples, payload.message)
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Ollama local call failed: {e}")

        store.append_history(payload.session_id, "user", payload.message)
        store.append_history(payload.session_id, "assistant", answer)
        return ChatResponse(session_id=payload.session_id, reply=answer, mode=payload.mode)
    # ===============================

    require_api_key()
    store.get_or_create(payload.session_id)
    history = store.get_history(payload.session_id)

    rag_context = ""
    if payload.use_rag:
         rag_context = make_rag_context(payload.session_id, payload.message)
    
    instructions = build_instructions(payload.mode, rag_context)

    model = get_gemini_model(system_instruction=instructions)
    gemini_history = _to_gemini_history(history)

    try:
        chat_session = model.start_chat(history=gemini_history)
        response = chat_session.send_message(payload.message)
        answer = response.text
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Gemini call failed: {e}")

    store.append_history(payload.session_id, "user", payload.message)
    store.append_history(payload.session_id, "assistant", answer)

    return ChatResponse(session_id=payload.session_id, reply=answer, mode=payload.mode)


@app.post("/api/chat/stream")
def chat_stream(payload: ChatRequest):
    if not payload.session_id or len(payload.session_id) < 6:
        payload.session_id = uuid.uuid4().hex
        store.get_or_create(payload.session_id)
    
    if not payload.message.strip():
        raise HTTPException(status_code=400, detail="Empty input.")
    
    if is_crisis(payload.message):
        store.get_or_create(payload.session_id)

        def gen():
            for i in range(0, len(CRISIS_RESPONSE), 12):
                yield CRISIS_RESPONSE[i : i + 12].encode("utf-8")
            store.append_history(payload.session_id, "user", payload.message)
            store.append_history(payload.session_id, "assistant", CRISIS_RESPONSE)

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
    # ===== LOCAL ONLY (Ollama) =====
    if is_local_only():
        store.get_or_create(payload.session_id)
        autoload_builtin_kb(payload.session_id)
        history = store.get_history(payload.session_id)

        rag_context = ""
        if payload.use_rag:
            rag_context = make_rag_context_local(payload.session_id, payload.message)

        instructions = build_instructions(payload.mode, rag_context)
        hist_tuples = [(m["role"], m.get("content", "")) for m in history if m.get("content")]

        try:
            text = local_generate_reply(payload.mode, instructions, hist_tuples, payload.message)
        except Exception as e:
            text = f"\n\n[Error] Ollama local call failed: {e}\n"

        def gen():
            for i in range(0, len(text), 10):
                yield text[i : i + 10].encode("utf-8")
            store.append_history(payload.session_id, "user", payload.message)
            store.append_history(payload.session_id, "assistant", text)

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")
    # ===============================

    if is_demo():
        store.get_or_create(payload.session_id)

        rag_context = ""
        if payload.use_rag:
            rag_context = make_rag_context_demo(payload.session_id, payload.message)

        text = demo_generate_reply(payload.session_id, payload.mode, payload.message)
        if rag_context:
            text += "\n\n---\nTop related notes (demo):\n" + rag_context

        def gen():
            for i in range(0, len(text), 10):
                yield text[i : i + 10].encode("utf-8")
            store.append_history(payload.session_id, "user", payload.message)
            store.append_history(payload.session_id, "assistant", text)

        return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

    require_api_key()
    store.get_or_create(payload.session_id)
    history = store.get_history(payload.session_id)

    rag_context = ""
    if payload.use_rag:
        rag_context = make_rag_context(payload.session_id, payload.message)
        
    instructions = build_instructions(payload.mode, rag_context)

    model = get_gemini_model(system_instruction=instructions)
    gemini_history = _to_gemini_history(history)

    def gen() -> Generator[bytes, None, None]:
        full_parts: List[str] = []
        try:
            chat_session = model.start_chat(history=gemini_history)
            response = chat_session.send_message(payload.message, stream=True)
            
            for chunk in response:
                if chunk.text:
                    full_parts.append(chunk.text)
                    yield chunk.text.encode("utf-8")
                    
        except Exception as e:
            msg = f"\n\n[Error] Gemini call failed: {e}\n"
            full_parts.append(msg)
            yield msg.encode("utf-8")
        finally:
            answer = "".join(full_parts).strip()
            store.append_history(payload.session_id, "user", payload.message)
            store.append_history(payload.session_id, "assistant", answer)

    return StreamingResponse(gen(), media_type="text/plain; charset=utf-8")

@app.get("/api/session/history")
def session_history(session_id: str = Query(...)):
    store.get_or_create(session_id)
    return {"session_id": session_id, "history": store.get_history(session_id)}