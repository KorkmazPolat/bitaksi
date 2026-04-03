"""
FastAPI dependency injection: shared service instances.
"""
from __future__ import annotations

from functools import lru_cache
from pathlib import Path

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import get_settings
from src.generation.response_generator import GenerationResult, ResponseGenerator
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.retriever import BaseRetriever
from src.retrieval.smart_grounding import SmartGroundingRetriever


class ChatService:
    """Orchestrates retrieval → generation for a chat query."""

    def __init__(
        self,
        grounding: SmartGroundingRetriever,
        generator: ResponseGenerator,
    ):
        self.grounding = grounding
        self.generator = generator

    def answer(
        self,
        query: str,
        history=None,
    ) -> GenerationResult:
        retrieval_result = self.grounding.retrieve(query)
        return self.generator.generate(
            query=query,
            retrieval_result=retrieval_result,
            history=history,
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

    return ChatService(grounding=grounding, generator=generator)


def get_chat_service() -> ChatService:
    return _build_service()
