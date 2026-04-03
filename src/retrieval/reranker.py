"""
Cross-Encoder Reranker
======================

After initial retrieval (dense + BM25), a cross-encoder re-scores every
(query, chunk) pair jointly — unlike bi-encoders that embed independently,
a cross-encoder sees both texts together and produces a highly accurate
relevance score.

Model: mmarco-mMiniLMv2-L12-H384 (multilingual, fine-tuned on MS MARCO).
Chosen because:
  - Supports Turkish without special tokenization
  - 12-layer MiniLM: fast enough for 20-50 candidates per query
  - Fine-tuned on passage retrieval, maps well to HR policy Q&A

Flow:
  1. Dense + BM25 retrieval: top-N candidates (default 20)
  2. Reranker: score all N pairs jointly
  3. Return top-K by reranker score (default 5)

The reranker score replaces the bi-encoder similarity score so downstream
components (grounding check, generation) always see the best signal.
"""
from __future__ import annotations

import logging
from functools import lru_cache

from src.config import get_settings
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def _load_cross_encoder(model_name: str):
    """Load cross-encoder once and cache globally."""
    try:
        from sentence_transformers import CrossEncoder
        logger.info("Loading cross-encoder: %s", model_name)
        return CrossEncoder(model_name, max_length=512)
    except Exception as exc:
        logger.warning("Cross-encoder load failed (%s): %s — reranking disabled", model_name, exc)
        return None


class CrossEncoderReranker:
    """
    Reranks retrieved chunks using a cross-encoder model.
    Falls back to original order if model unavailable.
    """

    def __init__(self):
        settings = get_settings()
        self.model_name = settings.rerank_model
        self.final_k = settings.rerank_final_k
        self._model = _load_cross_encoder(self.model_name)

    def rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
        top_k: int | None = None,
    ) -> list[RetrievedChunk]:
        """
        Score all (query, chunk) pairs and return top_k sorted by score.
        Falls back gracefully to original order if model unavailable.
        """
        k = top_k or self.final_k

        if not chunks:
            return []

        if self._model is None:
            logger.debug("Reranker unavailable — returning top-%d by original score", k)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:k]

        try:
            # Cross-encoder scores each (query, text) pair jointly
            # Use child chunk text for precise matching
            pairs = [(query, c.text) for c in chunks]
            scores: list[float] = self._model.predict(pairs).tolist()

            # Attach reranker score and sort
            scored = sorted(
                zip(scores, chunks),
                key=lambda x: x[0],
                reverse=True,
            )

            result = []
            for score, chunk in scored[:k]:
                # Replace bi-encoder score with calibrated cross-encoder score
                chunk.score = float(score)
                result.append(chunk)

            return result

        except Exception as exc:
            logger.warning("Reranking failed: %s — falling back to original order", exc)
            return sorted(chunks, key=lambda c: c.score, reverse=True)[:k]
