"""
BM25 Retriever
==============

Keyword-based retrieval that complements dense (embedding) search.
Dense search excels at semantic similarity; BM25 excels at exact-term
recall — policy codes, article numbers (e.g. "SGK Madde 4/a"), names,
specific numeric values.

Index:
  - Built lazily on first query from the ChromaDB raw collection.
  - Rebuilt automatically when new documents are ingested (via rebuild()).
  - Held in memory; no disk persistence needed (fast to rebuild from ChromaDB).

Tokenization:
  - Lowercase + whitespace split (handles Turkish characters correctly).
  - No stopword removal — policy terms are meaningful even when common.

Scoring:
  - BM25Okapi (rank-bm25 library) — standard BM25 with saturation k1=1.5, b=0.75.
  - Scores are normalised to [0, 1] range before returning.
"""
from __future__ import annotations

import logging
import re
from threading import Lock

from src.config import get_settings
from src.retrieval.retriever import RetrievedChunk

logger = logging.getLogger(__name__)

_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


def _tokenize(text: str) -> list[str]:
    """Lowercase unicode word tokenizer — preserves Turkish characters."""
    return _TOKEN_RE.findall(text.lower())


class BM25Retriever:
    """
    BM25 retriever backed by the ChromaDB raw collection.

    Usage:
        retriever = BM25Retriever(raw_collection)
        results = retriever.retrieve("SGK prim bildirimi", top_k=20)
    """

    def __init__(self, raw_collection):
        self._collection = raw_collection
        self._lock = Lock()
        self._bm25 = None
        self._chunks: list[RetrievedChunk] = []

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        """Return top_k chunks ranked by BM25 score."""
        settings = get_settings()
        k = top_k or settings.rerank_candidate_k

        self._ensure_index()

        if self._bm25 is None or not self._chunks:
            return []

        try:
            query_tokens = _tokenize(query)
            raw_scores = self._bm25.get_scores(query_tokens)

            # Normalise to [0, 1]
            max_score = max(raw_scores) if raw_scores.any() else 1.0
            if max_score == 0:
                return []
            norm_scores = raw_scores / max_score

            # Pair with chunks, filter zeros, sort, take top-k
            scored = [
                (float(s), chunk)
                for s, chunk in zip(norm_scores, self._chunks)
                if s > 0
            ]
            scored.sort(key=lambda x: x[0], reverse=True)

            result = []
            for score, chunk in scored[:k]:
                chunk.score = score
                result.append(chunk)
            return result

        except Exception as exc:
            logger.warning("BM25 retrieval failed: %s", exc)
            return []

    def rebuild(self) -> None:
        """Force a full index rebuild from ChromaDB (call after ingestion)."""
        with self._lock:
            self._bm25 = None
            self._chunks = []
        self._ensure_index()

    # ------------------------------------------------------------------
    # Index construction
    # ------------------------------------------------------------------

    def _ensure_index(self) -> None:
        """Build BM25 index lazily (once per process or after rebuild())."""
        if self._bm25 is not None:
            return

        with self._lock:
            if self._bm25 is not None:   # double-checked locking
                return
            self._build_index()

    def _build_index(self) -> None:
        try:
            from rank_bm25 import BM25Okapi
        except ImportError:
            logger.warning("rank-bm25 not installed — BM25 retrieval disabled")
            return

        try:
            logger.info("Building BM25 index from ChromaDB...")
            all_docs = self._collection.get(include=["documents", "metadatas"])

            docs: list[str] = all_docs.get("documents") or []
            metas: list[dict] = all_docs.get("metadatas") or []

            if not docs:
                logger.warning("BM25 index: no documents found in collection")
                return

            tokenized = [_tokenize(d) for d in docs]
            self._bm25 = BM25Okapi(tokenized, k1=1.5, b=0.75)
            self._chunks = [
                RetrievedChunk.from_metadata(doc, meta, score=0.0)
                for doc, meta in zip(docs, metas)
            ]
            logger.info("BM25 index built: %d documents", len(docs))

        except Exception as exc:
            logger.error("BM25 index build failed: %s", exc)
            self._bm25 = None
            self._chunks = []
