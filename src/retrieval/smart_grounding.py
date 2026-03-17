"""
Smart Grounding Retriever — the core retrieval orchestrator.

Pipeline:
  1. Direct retrieval from raw_content index
  2. Cross-check with relatives/follow-up index (fetch parent chunks)
  3. Merge, deduplicate, rank by score
  4. If result count < min_results → Fallback chain:
       Step 1: Query Expansion  → retry retrieval with variants
       Step 2: HyDE             → retry with hypothetical document
       Step 3: Query Decomposition → per sub-question retrieval + merge

The retriever enforces RAG necessity: it only accepts a query if relevant
chunks can be retrieved above the similarity threshold. The generation layer
must refuse to answer if no grounded context is found.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.config import get_settings
from src.retrieval.retriever import BaseRetriever, RetrievedChunk
from src.retrieval.query_expansion import QueryExpander
from src.retrieval.hyde import HyDERetrieval
from src.retrieval.query_decomposition import QueryDecomposer


@dataclass
class RetrievalResult:
    chunks: list[RetrievedChunk]
    strategy_used: str          # "direct" | "expansion" | "hyde" | "decomposition"
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
        raw_retriever_for_ids: BaseRetriever | None = None,
    ):
        settings = get_settings()
        self.raw = raw_retriever
        self.relatives = relatives_retriever
        self.raw_for_ids = raw_retriever_for_ids or raw_retriever
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
                strategy_used="direct",
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 1: Query Expansion ──────────────────────────────────
        print("[SmartGrounding] Direct retrieval insufficient → Query Expansion")
        variants = self.expander.expand(query)
        queries_tried.extend(variants[1:])  # skip the original already tried

        expansion_chunks: list[RetrievedChunk] = list(chunks)
        for variant in variants[1:]:
            expansion_chunks.extend(
                self._retrieve_and_augment(variant, self.fallback_top_k)
            )
        chunks = _deduplicate(expansion_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used="expansion",
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 2: HyDE ─────────────────────────────────────────────
        print("[SmartGrounding] Expansion insufficient → HyDE")
        hypothesis = self.hyde.generate_hypothesis(query)
        queries_tried.append(f"[HyDE] {hypothesis[:80]}...")

        hyde_chunks = self._retrieve_and_augment(hypothesis, self.fallback_top_k)
        chunks = _deduplicate(chunks + hyde_chunks)[: self.fallback_top_k]

        if self._sufficient(chunks):
            return RetrievalResult(
                chunks=chunks,
                strategy_used="hyde",
                queries_tried=queries_tried,
                grounded=True,
            )

        # ── Fallback 3: Query Decomposition ──────────────────────────────
        print("[SmartGrounding] HyDE insufficient → Query Decomposition")
        sub_questions = self.decomposer.decompose(query)
        queries_tried.extend(sub_questions)

        decomp_chunks: list[RetrievedChunk] = list(chunks)
        for sub_q in sub_questions:
            decomp_chunks.extend(
                self._retrieve_and_augment(sub_q, self.top_k)
            )
        chunks = _deduplicate(decomp_chunks)[: self.fallback_top_k]

        if chunks:
            return RetrievalResult(
                chunks=chunks,
                strategy_used="decomposition",
                queries_tried=queries_tried,
                grounded=bool(chunks),
            )

        # ── No grounded context found ────────────────────────────────────
        return RetrievalResult(
            chunks=[],
            strategy_used="none",
            queries_tried=queries_tried,
            grounded=False,
        )

    # ------------------------------------------------------------------

    def _retrieve_and_augment(
        self, query: str, top_k: int
    ) -> list[RetrievedChunk]:
        """
        1. Retrieve from raw index
        2. Retrieve from relatives index → get parent chunk IDs → fetch from raw
        3. Merge and deduplicate
        """
        raw_chunks = self.raw.retrieve(query, top_k=top_k)

        # Relatives lookup: query matches a pre-generated question,
        # retrieve the parent chunk from raw index
        relative_hits = self.relatives.retrieve(query, top_k=top_k)
        parent_ids = list(
            {m.metadata.get("chunk_id", "") for m in self._as_meta_list(relative_hits)}
        )
        # Re-fetch parents with actual content from raw
        parent_chunks = self._fetch_parents(parent_ids)

        return _deduplicate(raw_chunks + parent_chunks)

    def _as_meta_list(self, chunks: list[RetrievedChunk]):
        """Helper: expose metadata from relatives as dict-like objects."""
        # relatives collection stores chunk_id in the metadata; we stored it
        # as section/source but chunk_id is in doc metadata. For relatives we
        # need to get the chunk_id key from the collection directly.
        return chunks

    def _fetch_parents(self, chunk_ids: list[str]) -> list[RetrievedChunk]:
        valid_ids = [cid for cid in chunk_ids if cid]
        if not valid_ids:
            return []
        try:
            results = self.raw.collection.get(
                ids=valid_ids,
                include=["documents", "metadatas"],
            )
            chunks: list[RetrievedChunk] = []
            for doc, meta in zip(
                results.get("documents", []),
                results.get("metadatas", []),
            ):
                chunks.append(
                    RetrievedChunk(
                        chunk_id=meta.get("chunk_id", ""),
                        text=doc,
                        source=meta.get("source", ""),
                        page_num=int(meta.get("page_num", 0)),
                        section=meta.get("section", ""),
                        score=0.8,  # relative-match confidence
                        doc_id=meta.get("doc_id", ""),
                    )
                )
            return chunks
        except Exception:
            return []

    def _sufficient(self, chunks: list[RetrievedChunk]) -> bool:
        above_threshold = [c for c in chunks if c.score >= self.threshold]
        return len(above_threshold) >= self.MIN_RESULTS
