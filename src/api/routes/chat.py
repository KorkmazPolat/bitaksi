"""
Chat routes: the main employee-facing chat endpoint.
"""
from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, field_validator

from src.api.dependencies import get_chat_debug_store, get_chat_service
from src.config import get_settings
from src.generation.response_generator import ChatMessage
from src.utils.enums import MessageRole

router = APIRouter(prefix="/chat", tags=["chat"])


class ChatRequest(BaseModel):
    query: str
    session_id: str = "default"
    history: list[dict] = []

    @field_validator("query")
    @classmethod
    def validate_query(cls, v: str) -> str:
        v = v.strip()
        if not v:
            raise ValueError("Query cannot be empty.")
        max_len = get_settings().max_query_length
        if len(v) > max_len:
            raise ValueError(f"Query exceeds maximum length of {max_len} characters.")
        return v


class SourceRef(BaseModel):
    document: str
    page: int
    section: str
    score: float
    chunk_text: str = ""   # sent to frontend for text highlighting
    highlight_text: str = ""


class CitationRef(BaseModel):
    document: str
    page: int
    context: str = ""
    source_index: int = -1
    evidence_text: str = ""


class ChatResponse(BaseModel):
    answer: str
    sources: list[SourceRef]
    citations: list[CitationRef]
    strategy_used: str
    grounded: bool
    answer_type: str
    queries_tried: list[str]


class DebugChunkRef(BaseModel):
    chunk_id: str
    page: int
    section: str
    score: float
    document: str
    preview: str
    content_type: str


class ChatDebugItem(BaseModel):
    created_at: float
    query: str
    answer: str
    grounded: bool
    answer_type: str
    strategy_used: str
    queries_tried: list[str]
    sources: list[SourceRef]
    citations: list[CitationRef]
    retrieved_chunks: list[DebugChunkRef]
    trace_steps: list[dict]


class ChatSimulationResponse(ChatResponse):
    retrieved_chunks: list[DebugChunkRef]
    trace_steps: list[dict]
    threshold: float


@router.post("/", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    service=Depends(get_chat_service),
):
    valid_roles = {MessageRole.USER, MessageRole.ASSISTANT}
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in request.history
        if m.get("role") in valid_roles
    ]

    result = service.answer(query=request.query, history=history)

    return ChatResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        citations=[CitationRef(**c) for c in result.citations],
        strategy_used=result.strategy_used,
        grounded=result.grounded,
        answer_type=result.answer_type,
        queries_tried=result.queries_tried,
    )


@router.post("/simulate", response_model=ChatSimulationResponse)
async def simulate_chat(
    request: ChatRequest,
    service=Depends(get_chat_service),
):
    valid_roles = {MessageRole.USER, MessageRole.ASSISTANT}
    history = [
        ChatMessage(role=m["role"], content=m["content"])
        for m in request.history
        if m.get("role") in valid_roles
    ]

    retrieval_result = service.grounding.retrieve(request.query)
    result = service.generator.generate(
        query=request.query,
        retrieval_result=retrieval_result,
        history=history,
    )
    service.debug_store.add(
        service._build_debug_record(request.query, retrieval_result, result)
    )

    return ChatSimulationResponse(
        answer=result.answer,
        sources=[SourceRef(**s) for s in result.sources],
        citations=[CitationRef(**c) for c in result.citations],
        strategy_used=result.strategy_used,
        grounded=result.grounded,
        answer_type=result.answer_type,
        queries_tried=result.queries_tried,
        retrieved_chunks=[
            DebugChunkRef(
                chunk_id=chunk.chunk_id,
                page=chunk.page_num,
                section=chunk.section,
                score=round(chunk.score, 3),
                document=chunk.doc_id,
                preview=(chunk.text or "")[:280],
                content_type=str(chunk.content_type),
            )
            for chunk in retrieval_result.chunks
        ],
        trace_steps=list(retrieval_result.trace_steps),
        threshold=retrieval_result.threshold,
    )


@router.get("/debug/recent", response_model=list[ChatDebugItem])
async def recent_chat_debug(debug_store=Depends(get_chat_debug_store)):
    items = debug_store.list()
    return [
        ChatDebugItem(
            created_at=item["created_at"],
            query=item["query"],
            answer=item["answer"],
            grounded=item["grounded"],
            answer_type=item.get("answer_type", "answer"),
            strategy_used=item["strategy_used"],
            queries_tried=item["queries_tried"],
            sources=[SourceRef(**source) for source in item["sources"]],
            citations=[CitationRef(**citation) for citation in item["citations"]],
            retrieved_chunks=[
                DebugChunkRef(**chunk) for chunk in item["retrieved_chunks"]
            ],
            trace_steps=item.get("trace_steps", []),
        )
        for item in items
    ]
