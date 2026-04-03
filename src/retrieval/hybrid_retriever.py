"""
Hybrid Retriever
================

Combines dense (embedding) and sparse (BM25) retrieval with Reciprocal
Rank Fusion (RRF), then reranks the fused candidates with a cross-encoder.

Pipeline per query:
  ┌──────────────┐   ┌──────────────┐
  │ Dense search │   │  BM25 search │   ← parallel, top-N each
  └──────┬───────┘   └──────┬───────┘
         │                  │
         └──── RRF Fusion ──┘            ← combine without needing score calibration
                    │
           Deduplicate + sort
                    │
          Cross-Encoder Rerank           ← joint (query, chunk) scoring
                    │
           Parent text expansion         ← attach full section for generation
                    │
              top-K results
"""
from __future__ import annotations

import logging
from collections import defaultdict

from src.config import get_settings
from src.retrieval.retriever import BaseRetriever, RetrievedChunk
from src.retrieval.bm25_retriever import BM25Retriever
from src.retrieval.reranker import CrossEncoderReranker

logger = logging.getLogger(__name__)


def _rrf_score(rank: int, k: int) -> float:
    """Reciprocal Rank Fusion score for a given rank (0-indexed)."""
    return 1.0 / (k + rank + 1)


def _rrf_fuse(
    dense_results: list[RetrievedChunk],
    bm25_results: list[RetrievedChunk],
    rrf_k: int,
) -> list[RetrievedChunk]:
    """
    Merge two ranked lists with RRF.

    RRF is robust because it combines ranks, not raw scores, so there is
    no need to calibrate dense cosine similarity against BM25 scores.
    Both lists contribute equally.
    """
    # Accumulate RRF scores per chunk_id
    rrf_scores: dict[str, float] = defaultdict(float)
    chunk_by_id: dict[str, RetrievedChunk] = {}

    for rank, chunk in enumerate(dense_results):
        rrf_scores[chunk.chunk_id] += _rrf_score(rank, rrf_k)
        chunk_by_id[chunk.chunk_id] = chunk

    for rank, chunk in enumerate(bm25_results):
        rrf_scores[chunk.chunk_id] += _rrf_score(rank, rrf_k)
        if chunk.chunk_id not in chunk_by_id:
            chunk_by_id[chunk.chunk_id] = chunk

    # Sort by fused score, attach as the chunk's score
    fused: list[RetrievedChunk] = []
    for chunk_id, score in sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True):
        chunk = chunk_by_id[chunk_id]
        chunk.score = score
        fused.append(chunk)

    return fused


class HybridRetriever:
    """
    Dense + BM25 + RRF + Cross-Encoder Reranker in a single callable.

    Both BM25 and reranking are feature-flagged (use_bm25, use_reranking)
    so the system degrades gracefully when dependencies are unavailable.
    """

    def __init__(
        self,
        dense_retriever: BaseRetriever,
        bm25_retriever: BM25Retriever,
        reranker: CrossEncoderReranker,
    ):
        settings = get_settings()
        self.dense = dense_retriever
        self.bm25 = bm25_retriever
        self.reranker = reranker
        self.use_bm25 = settings.use_bm25
        self.use_reranking = settings.use_reranking
        self.candidate_k = settings.rerank_candidate_k
        self.final_k = settings.rerank_final_k
        self.rrf_k = settings.rrf_k

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """
        Full hybrid pipeline: dense → BM25 → RRF → rerank → parent expand.
        top_k overrides rerank_final_k when provided.
        """
        k = top_k or self.final_k
        candidate_k = max(self.candidate_k, k * 3)

        # ── Dense retrieval ───────────────────────────────────────────
        dense_results = self.dense.retrieve(query, top_k=candidate_k)

        # ── BM25 retrieval (optional) ─────────────────────────────────
        if self.use_bm25:
            bm25_results = self.bm25.retrieve(query, top_k=candidate_k)
            candidates = _rrf_fuse(dense_results, bm25_results, self.rrf_k)
        else:
            candidates = dense_results

        if not candidates:
            return []

        # ── Cross-encoder rerank (optional) ──────────────────────────
        if self.use_reranking:
            ranked = self.reranker.rerank(query, candidates, top_k=k)
        else:
            ranked = sorted(candidates, key=lambda c: c.score, reverse=True)[:k]

        # ── Parent text expansion ─────────────────────────────────────
        # parent_text already populated from ChromaDB metadata in from_metadata()
        # Nothing extra needed here — generation layer reads chunk.parent_text

        return ranked

    def invalidate_bm25(self) -> None:
        """Call after ingesting new documents to rebuild the BM25 index."""
        self.bm25.rebuild()
