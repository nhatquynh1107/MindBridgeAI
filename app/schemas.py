from __future__ import annotations

from pydantic import BaseModel, Field


class ChatRequest(BaseModel):
    session_id: str = Field(..., min_length=6)
    message: str = Field(..., min_length=1)
    mode: str = Field(default="Health")
    use_rag: bool = Field(default=True)

class ChatResponse(BaseModel):
    session_id: str
    reply: str
    mode: str

class NewSessionResponse(BaseModel):
    session_id: str

class ClearRequest(BaseModel):
    session_id: str = Field(..., min_length=6)
