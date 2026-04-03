"""
FastAPI dependency injection: shared service instances.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from collections import deque
from dataclasses import asdict, dataclass, field
from threading import Lock
from time import time

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import get_settings
from src.generation.response_generator import GenerationResult, ResponseGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import BaseRetriever
from src.retrieval.smart_grounding import RetrievalResult, SmartGroundingRetriever


@dataclass
class DebugChunkRecord:
    chunk_id: str
    page: int
    section: str
    score: float
    document: str
    preview: str
    content_type: str


@dataclass
class ChatDebugRecord:
    created_at: float
    query: str
    answer: str
    grounded: bool
    strategy_used: str
    queries_tried: list[str] = field(default_factory=list)
    sources: list[dict] = field(default_factory=list)
    citations: list[dict] = field(default_factory=list)
    retrieved_chunks: list[DebugChunkRecord] = field(default_factory=list)


class ChatDebugStore:
    def __init__(self, limit: int = 100):
        self._items = deque(maxlen=limit)
        self._lock = Lock()

    def add(self, record: ChatDebugRecord) -> None:
        with self._lock:
            self._items.appendleft(record)

    def list(self) -> list[dict]:
        with self._lock:
            return [
                {
                    **asdict(item),
                    "retrieved_chunks": [asdict(chunk) for chunk in item.retrieved_chunks],
                }
                for item in self._items
            ]


class ChatService:
    """Orchestrates retrieval → generation for a chat query."""

    def __init__(
        self,
        grounding: SmartGroundingRetriever,
        generator: ResponseGenerator,
        debug_store: ChatDebugStore,
    ):
        self.grounding = grounding
        self.generator = generator
        self.debug_store = debug_store

    def answer(
        self,
        query: str,
        history=None,
    ) -> GenerationResult:
        retrieval_result = self.grounding.retrieve(query)
        result = self.generator.generate(
            query=query,
            retrieval_result=retrieval_result,
            history=history,
        )
        self.debug_store.add(self._build_debug_record(query, retrieval_result, result))
        return result

    @staticmethod
    def _build_debug_record(
        query: str,
        retrieval_result: RetrievalResult,
        result: GenerationResult,
    ) -> ChatDebugRecord:
        return ChatDebugRecord(
            created_at=time(),
            query=query,
            answer=result.answer,
            grounded=result.grounded,
            strategy_used=result.strategy_used,
            queries_tried=list(result.queries_tried),
            sources=list(result.sources),
            citations=list(result.citations),
            retrieved_chunks=[
                DebugChunkRecord(
                    chunk_id=chunk.chunk_id,
                    page=chunk.page_num,
                    section=chunk.section,
                    score=round(chunk.score, 3),
                    document=Path(chunk.source).stem if chunk.source else "",
                    preview=(chunk.text or "")[:280],
                    content_type=str(chunk.content_type),
                )
                for chunk in retrieval_result.chunks
            ],
        )


@lru_cache(maxsize=1)
def _build_service() -> ChatService:
    settings = get_settings()
    embedding_fn = SentenceTransformerEmbeddingFunction(
        model_name=settings.embedding_model
    )

    Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
    chroma_client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

    raw_col = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_raw,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )
    relatives_col = chroma_client.get_or_create_collection(
        name=settings.chroma_collection_relatives,
        embedding_function=embedding_fn,
        metadata={"hnsw:space": "cosine"},
    )

    # ── Retrieval stack ───────────────────────────────────────────────
    dense_retriever = BaseRetriever(
        collection=raw_col,
        top_k=settings.top_k,
        score_threshold=settings.similarity_threshold,
    )
    bm25_retriever = BM25Retriever(raw_collection=raw_col)
    reranker = CrossEncoderReranker()

    hybrid_retriever = HybridRetriever(
        dense_retriever=dense_retriever,
        bm25_retriever=bm25_retriever,
        reranker=reranker,
    )

    relatives_retriever = BaseRetriever(
        collection=relatives_col,
        top_k=settings.top_k,
        score_threshold=settings.relatives_score_threshold,
    )

    grounding = SmartGroundingRetriever(
        hybrid_retriever=hybrid_retriever,
        relatives_retriever=relatives_retriever,
    )
    generator = ResponseGenerator()
    debug_store = ChatDebugStore(limit=120)

    return ChatService(
        grounding=grounding,
        generator=generator,
        debug_store=debug_store,
    )


def get_chat_service() -> ChatService:
    return _build_service()


def get_chat_debug_store() -> ChatDebugStore:
    return _build_service().debug_store
