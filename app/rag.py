from __future__ import annotations

import re
from typing import List, Tuple

import numpy as np

try:
    from pypdf import PdfReader  # optional
except Exception:
    PdfReader = None


def _clean_text(s: str) -> str:
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def chunk_by_words(text: str, chunk_words: int = 220, overlap_words: int = 40) -> List[str]:
    """Simple word-based chunking with overlap (good enough for MVP)."""
    words = text.split()
    if not words:
        return []
    out: List[str] = []
    i = 0
    while i < len(words):
        j = min(len(words), i + chunk_words)
        out.append(" ".join(words[i:j]))
        if j >= len(words):
            break
        i = max(0, j - overlap_words)
    return out


def cosine_top_k(query_vec: List[float], items: List[Tuple[str, List[float], str]], k: int = 4):
    """Return top-k by cosine similarity."""
    if not items:
        return []
    q = np.array(query_vec, dtype=np.float32)
    qn = float(np.linalg.norm(q) + 1e-9)

    scored = []
    for chunk_id, emb, text in items:
        v = np.array(emb, dtype=np.float32)
        score = float(np.dot(q, v) / (qn * (np.linalg.norm(v) + 1e-9)))
        scored.append((chunk_id, score, text))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:k]


def read_text_file(filename: str, raw: bytes) -> str:
    for enc in ("utf-8", "utf-8-sig", "latin-1"):
        try:
            return _clean_text(raw.decode(enc))
        except Exception:
            pass
    return _clean_text(raw.decode("utf-8", errors="ignore"))


def read_pdf_file(raw: bytes) -> str:
    if PdfReader is None:
        raise RuntimeError("PDF support missing. Install pypdf to enable PDF upload.")
    import io
    reader = PdfReader(io.BytesIO(raw))
    pages = []
    for p in reader.pages:
        pages.append(p.extract_text() or "")
    return _clean_text("\n".join(pages))
