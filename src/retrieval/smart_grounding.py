"""
Smart Grounding Retriever — the core retrieval orchestrator.

Each retrieval attempt now goes through the full HybridRetriever pipeline:
  Dense (ChromaDB) + BM25 → RRF fusion → Cross-encoder rerank → Parent expand

Fallback chain (triggered when results are insufficient):
  Direct → Query Expansion → HyDE → Query Decomposition

The relatives (Q&A) index augments every retrieval attempt.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from src.config import get_settings
from src.retrieval.hybrid_retriever import HybridRetriever
from src.retrieval.retriever import BaseRetriever, RetrievedChunk
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hyde import HyDERetrieval
from src.retrieval.query_decomposition import QueryDecomposer
from src.utils.enums import RetrievalStrategy

logger = logging.getLogger(__name__)
_TOKEN_RE = re.compile(r"\w+", re.UNICODE)


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    strategy_used: RetrievalStrategy
    queries_tried: list[str] = field(default_factory=list)
    grounded: bool = True


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
    Orchestrates hybrid retrieval + fallback chain.
    Ensures every answer is grounded in retrieved document evidence.
    """

    MIN_RESULTS = 1

    def __init__(
        self,
        hybrid_retriever: HybridRetriever,
        relatives_retriever: BaseRetriever,
    ):
        settings = get_settings()
        self.hybrid = hybrid_retriever
        self.relatives = relatives_retriever
        self.threshold = settings.similarity_threshold
        self.top_k = settings.top_k
        self.fallback_top_k = settings.fallback_top_k
        lexical_overlap_weight = getattr(settings, "lexical_overlap_weight", 0.15)
        self.lexical_overlap_weight = (
            float(lexical_overlap_weight)
            if isinstance(lexical_overlap_weight, (int, float))
            else 0.15
        )

        self.expander = QueryExpander()
        self.hyde = HyDERetrieval()
        self.decomposer = QueryDecomposer()

    def retrieve(self, query: str) -> RetrievalResult:
        queries_tried = [query]

        # ── Direct hybrid retrieval ───────────────────────────────────
        chunks = self._retrieve_and_augment(query, self.top_k)
        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks, strategy_used=RetrievalStrategy.DIRECT,
                queries_tried=queries_tried, grounded=True,
            )

        # ── Fallback 1: Query Expansion ──────────────────────────────
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
                chunks=chunks, strategy_used=RetrievalStrategy.EXPANSION,
                queries_tried=queries_tried, grounded=True,
            )

        # ── Fallback 2: HyDE ─────────────────────────────────────────
        logger.info("Expansion insufficient → HyDE")
        hypothesis = self.hyde.generate_hypothesis(query)
        queries_tried.append(f"[HyDE] {hypothesis[:80]}...")

        hyde_chunks = self._retrieve_and_augment(hypothesis, self.fallback_top_k)
        chunks = _deduplicate(chunks + hyde_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks, strategy_used=RetrievalStrategy.HYDE,
                queries_tried=queries_tried, grounded=True,
            )

        # ── Fallback 3: Query Decomposition ──────────────────────────
        logger.info("HyDE insufficient → Query Decomposition")
        sub_questions = self.decomposer.decompose(query)
        queries_tried.extend(sub_questions)

        decomp_chunks: list[RetrievedChunk] = list(chunks)
        for sub_q in sub_questions:
            decomp_chunks.extend(self._retrieve_and_augment(sub_q, self.top_k))
        chunks = _deduplicate(decomp_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.DECOMPOSITION,
                queries_tried=queries_tried,
                grounded=True,
            )

        return RetrievalResult(
            chunks=[], strategy_used=RetrievalStrategy.NONE,
            queries_tried=queries_tried, grounded=False,
        )

    # ------------------------------------------------------------------

    def _retrieve_and_augment(self, query: str, top_k: int) -> list[RetrievedChunk]:
        """Hybrid retrieval + relatives Q&A index augmentation."""
        # Main hybrid pipeline (dense + BM25 + rerank)
        raw_chunks = self.hybrid.retrieve(query, top_k=top_k)

        # Relatives lookup: pre-generated questions → parent chunks
        relative_hits = self.relatives.retrieve(query, top_k=top_k)
        parent_ids = self._parent_ids_from_hits(relative_hits)
        parent_score_map = self._parent_score_map(relative_hits)
        parent_chunks = self.hybrid.dense.fetch_by_ids(parent_ids)

        # Apply conservative scores from score_map to fetched parent chunks
        for chunk in parent_chunks:
            if chunk.chunk_id in parent_score_map:
                chunk.score = parent_score_map[chunk.chunk_id]

        merged = _deduplicate(raw_chunks + parent_chunks)
        return self._rerank(query, merged)

    @staticmethod
    def _parent_ids_from_hits(hits: list[RetrievedChunk]) -> list[str]:
        seen: set[str] = set()
        ids: list[str] = []
        for hit in hits:
            cid = hit.chunk_id
            if cid and cid not in seen:
                seen.add(cid)
                ids.append(cid)
        return ids

    @staticmethod
    def _parent_score_map(hits: list[RetrievedChunk]) -> dict[str, float]:
        """Assign parent chunks a conservative score from the relatives hit."""
        out: dict[str, float] = {}
        for hit in hits:
            cid = hit.chunk_id
            if not cid:
                continue
            score = max(0.0, min(1.0, float(hit.score) * 0.9))
            if score > out.get(cid, 0.0):
                out[cid] = score
        return out

    def _sufficient(self, chunks: list[RetrievedChunk]) -> bool:
        above = [c for c in chunks if c.score >= self.threshold]
        return len(above) >= self.MIN_RESULTS

    def _rerank(
        self,
        query: str,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        query_terms = self._normalized_terms(query)
        ranked: list[RetrievedChunk] = []
        for chunk in chunks:
            text_terms = self._normalized_terms(
                f"{chunk.section}\n{chunk.text[:400]}"
            )
            overlap = (
                len(query_terms & text_terms) / max(len(query_terms), 1)
                if query_terms else 0.0
            )
            adjusted_score = max(
                0.0,
                min(1.0, chunk.score + (overlap * self.lexical_overlap_weight))
            )
            ranked.append(
                RetrievedChunk(
                    chunk_id=chunk.chunk_id,
                    text=chunk.text,
                    source=chunk.source,
                    page_num=chunk.page_num,
                    section=chunk.section,
                    score=adjusted_score,
                    doc_id=chunk.doc_id,
                    content_type=chunk.content_type,
                    parent_text=chunk.parent_text,
                    breadcrumb=chunk.breadcrumb,
                )
            )
        return sorted(ranked, key=lambda c: c.score, reverse=True)

    @staticmethod
    def _normalized_terms(text: str) -> set[str]:
        return {
            tok.casefold()
            for tok in _TOKEN_RE.findall(text or "")
            if len(tok) >= 4
        }
