"""
Smart Grounding Retriever — the core retrieval orchestrator.

Pipeline:
  1. Direct retrieval from raw_content index
  2. Cross-check with relatives/follow-up index (fetch parent chunks)
  3. Merge, deduplicate, rank by score
  4. If result count < MIN_RESULTS → Fallback chain:
       Step 1: Query Expansion  → retry retrieval with variants
       Step 2: HyDE             → retry with hypothetical document
       Step 3: Query Decomposition → per sub-question retrieval + merge

The retriever enforces RAG necessity: it only accepts a query if relevant
chunks can be retrieved above the similarity threshold. The generation layer
must refuse to answer if no grounded context is found.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

from src.config import get_settings
from src.retrieval.retriever import BaseRetriever, RetrievedChunk
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hyde import HyDERetrieval
from src.retrieval.query_decomposition import QueryDecomposer
from src.utils.enums import RetrievalStrategy

logger = logging.getLogger(__name__)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    strategy_used: RetrievalStrategy
    queries_tried: list[str] = field(default_factory=list)
    grounded: bool = True       # False if no chunks above threshold found


def _deduplicate(chunks: list[RetrievedChunk]) -> list[RetrievedChunk]:
    seen: set[str] = set()
    unique: list[RetrievedChunk] = []
    for c in sorted(chunks, key=lambda x: x.score, reverse=True):
        if c.chunk_id not in seen:
            seen.add(c.chunk_id)
            unique.append(c)
    return unique


class SmartGroundingRetriever:
    """
    Orchestrates the full retrieval pipeline with fallback strategies.
    Ensures every answer is grounded in retrieved document evidence.
    """

    MIN_RESULTS = 2

    def __init__(
        self,
        raw_retriever: BaseRetriever,
        relatives_retriever: BaseRetriever,
    ):
        settings = get_settings()
        self.raw = raw_retriever
        self.relatives = relatives_retriever
        self.threshold = settings.similarity_threshold
        self.top_k = settings.top_k
        self.fallback_top_k = settings.fallback_top_k

        self.expander = QueryExpander()
        self.hyde = HyDERetrieval()
        self.decomposer = QueryDecomposer()

    def retrieve(self, query: str) -> RetrievalResult:
        queries_tried = [query]

        # ── Step 0: Direct retrieval ─────────────────────────────────────
        chunks = self._retrieve_and_augment(query, self.top_k)
        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.DIRECT,
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 1: Query Expansion ──────────────────────────────────
        logger.info("Direct retrieval insufficient → Query Expansion")
        variants = self.expander.expand(query)
        queries_tried.extend(variants[1:])

        expansion_chunks: list[RetrievedChunk] = list(chunks)
        for variant in variants[1:]:
            expansion_chunks.extend(
                self._retrieve_and_augment(variant, self.fallback_top_k)
            )
        chunks = _deduplicate(expansion_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.EXPANSION,
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 2: HyDE ─────────────────────────────────────────────
        logger.info("Expansion insufficient → HyDE")
        hypothesis = self.hyde.generate_hypothesis(query)
        queries_tried.append(f"[HyDE] {hypothesis[:80]}...")

        hyde_chunks = self._retrieve_and_augment(hypothesis, self.fallback_top_k)
        chunks = _deduplicate(chunks + hyde_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.HYDE,
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 3: Query Decomposition ──────────────────────────────
        logger.info("HyDE insufficient → Query Decomposition")
        sub_questions = self.decomposer.decompose(query)
        queries_tried.extend(sub_questions)

        decomp_chunks: list[RetrievedChunk] = list(chunks)
        for sub_q in sub_questions:
            decomp_chunks.extend(self._retrieve_and_augment(sub_q, self.top_k))
        chunks = _deduplicate(decomp_chunks)[: self.fallback_top_k]

        if chunks:
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.DECOMPOSITION,
                queries_tried=queries_tried,
                grounded=bool(chunks),
            )

        # ── No grounded context found ────────────────────────────────────
        return RetrievalResult(
            chunks=[],
            strategy_used=RetrievalStrategy.NONE,
            queries_tried=queries_tried,
            grounded=False,
        )

    # ------------------------------------------------------------------

    def _retrieve_and_augment(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """
        1. Retrieve from raw index
        2. Retrieve from relatives index → collect parent chunk IDs → fetch from raw
        3. Merge and deduplicate
        """
        raw_chunks = self.raw.retrieve(query, top_k=top_k)

        # Relatives lookup: a matched question → its parent chunk ID
        relative_hits = self.relatives.retrieve(query, top_k=top_k)
        parent_ids = list({
            meta.get("chunk_id", "")
            for hit in relative_hits
            for meta in [{}]   # relatives metadata stored in collection.get
        })
        # Fetch parent chunk IDs from the relatives hit metadata directly
        parent_ids = self._parent_ids_from_hits(relative_hits)
        parent_chunks = self.raw.fetch_by_ids(parent_ids)

        return _deduplicate(raw_chunks + parent_chunks)

    @staticmethod
    def _parent_ids_from_hits(hits: list[RetrievedChunk]) -> list[str]:
        """
        Relatives chunks carry the parent raw chunk_id in their own chunk_id field
        (stored as "q_<hash>" in the collection with metadata.chunk_id pointing back).
        We stored it in metadata["chunk_id"] at index time.
        Since BaseRetriever.from_metadata reads chunk_id from meta["chunk_id"],
        and relatives records store the parent id there, we can read it directly.
        """
        seen: set[str] = set()
        ids: list[str] = []
        for hit in hits:
            cid = hit.chunk_id
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids

    def _sufficient(self, chunks: list[RetrievedChunk]) -> bool:
        above = [c for c in chunks if c.score >= self.threshold]
        return len(above) >= self.MIN_RESULTS
