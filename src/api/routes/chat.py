"""
Chat routes: the main employee-facing chat endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel

from src.api.dependencies import get_chat_service
from src.generation.response_generator import ChatMessage

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    history: list[dict] = []  # [{"role": "user"|"assistant", "content": "..."}]


class SourceRef(BaseModel):
    document: str
    page: int
    section: str
    score: float


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    strategy_used: str
    grounded: bool
    queries_tried: list[str]


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service=Depends(get_chat_service),
):
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in request.history
        if m.get("role") in {"user", "assistant"}
    ]

    result = service.answer(query=request.query, history=history)

    return ChatResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        strategy_used=result.strategy_used,
        grounded=result.grounded,
        queries_tried=result.queries_tried,
    )
