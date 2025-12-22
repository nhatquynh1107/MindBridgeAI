from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Dict, List


MAX_HISTORY_MESSAGES = 30  # keep last N messages to avoid memory blow


@dataclass
class RAGChunk:
    doc_name: str
    chunk_id: str  # e.g., "notes.md#3"
    text: str
    embedding: List[float]


@dataclass
class SessionData:
    history: List[Dict[str, str]] = field(default_factory=list)  # [{role, content}]
    rag_chunks: List[RAGChunk] = field(default_factory=list)


class InMemoryStore:
    """Thread-safe in-RAM store for demo/hackathon."""
    def __init__(self) -> None:
        self._lock = Lock()
        self._sessions: Dict[str, SessionData] = {}

    def get_or_create(self, session_id: str) -> SessionData:
        with self._lock:
            if session_id not in self._sessions:
                self._sessions[session_id] = SessionData()
            return self._sessions[session_id]

    def clear(self, session_id: str) -> None:
        with self._lock:
            self._sessions[session_id] = SessionData()

    def append_history(self, session_id: str, role: str, content: str) -> None:
        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData())
            s.history.append({"role": role, "content": content})
            if len(s.history) > MAX_HISTORY_MESSAGES:
                s.history = s.history[-MAX_HISTORY_MESSAGES:]

    def get_history(self, session_id: str) -> List[Dict[str, str]]:
        with self._lock:
            return list(self._sessions.get(session_id, SessionData()).history)

    def add_rag_chunks(self, session_id: str, chunks: List[RAGChunk]) -> None:
        with self._lock:
            s = self._sessions.setdefault(session_id, SessionData())
            s.rag_chunks.extend(chunks)

    def get_rag_chunks(self, session_id: str) -> List[RAGChunk]:
        with self._lock:
            return list(self._sessions.get(session_id, SessionData()).rag_chunks)

    def rag_status(self, session_id: str) -> Dict[str, Any]:
        with self._lock:
            s = self._sessions.get(session_id, SessionData())
            by_doc: Dict[str, int] = {}
            for c in s.rag_chunks:
                by_doc[c.doc_name] = by_doc.get(c.doc_name, 0) + 1
            return {"total_chunks": len(s.rag_chunks), "by_doc": by_doc}


store = InMemoryStore()
