# MindBridge AI

> **TechSprint Hackathon Submission**
> *Bridging the gap between teenage challenges and actionable solutions.*


## About The Project

Growing up is complicated. **MindBridge AI** is a lightweight, empathetic chatbot designed to act as an intelligent anchor for teenagers. It moves beyond generic chat to provide context-aware support for the specific stressors of modern adolescence.

Built for the **TechSprint Hackathon**, this MVP focuses on privacy, speed, and accessibility. It runs entirely locally with a high-performance **FastAPI** backend and a dependency-free **Vanilla JS** frontend.

## Key Features

### Specialized Support Modes
The AI adapts its persona and logic based on the user's selected mode:
* **Health & Wellness:** A calm space for emotional grounding (Non-diagnostic).
* **School Stress:** Prioritizes deadlines and breaks down homework into steps.
* **Friends & Social:** Generates scripts for conflict resolution and boundaries.
* **Coding Companion:** A collaborative debugger that explains logic.

### Technical Capabilities
* **Streaming Responses:** Real-time text generation for a natural feel.
* **On-Device RAG:** Upload documents (.txt/.pdf) for in-memory chunking and context-aware answers using cosine similarity.
* **Privacy-First:** Sessions and vector stores live in RAM; nothing is persisted to a database.
* **Demo Mode:** A built-in mock mode to run the UI without an API key or billing usage.

## Repository Structure

```text
ai-chatbot/
├── app/
│   ├── main.py          # FastAPI app + routes (chat, stream, sessions, RAG upload)
│   ├── prompts.py       # Mode list + system prompts per mode
│   ├── rag.py           # Chunking + cosine similarity helpers (+ PDF/TXT reading)
│   ├── schemas.py       # Pydantic request/response models
│   └── store.py         # In-memory session store (history + RAG chunks)
├── static/
│   ├── index.html       # UI layout
│   ├── styles.css       # UI styles
│   └── app.js           # Frontend logic (sessions, modes, chat, streaming)
├── requirements.txt     # Python dependencies
├── Dockerfile           # Container definition
├── docker-compose.yml   # Orchestration config
└── README.md            # Documentation


## Run locally (WSL/Linux/macOS)

### 1 Create venv + install deps

python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

### 2 Run in DeMO mode
export DEMO_MOCK=1
uvicorn app.main:app --host 127.0.0.1 --port 8000
# MindBridgeAI
