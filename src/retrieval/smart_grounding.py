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
from dataclasses import asdict, dataclass, field

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
    threshold: float = 0.0
    trace_steps: list[dict] = field(default_factory=list)


@dataclass
class RetrievalTraceChunk:
    chunk_id: str
    document: str
    page: int
    section: str
    score: float
    content_type: str
    preview: str


@dataclass
class RetrievalTraceStep:
    key: str
    title: str
    strategy: str
    status: str
    query: str = ""
    queries: list[str] = field(default_factory=list)
    reason: str = ""
    threshold: float = 0.0
    grounded_hits: int = 0
    chunks: list[RetrievalTraceChunk] = field(default_factory=list)
    substeps: list[dict] = field(default_factory=list)

    def as_dict(self) -> dict:
        return {
            **asdict(self),
            "chunks": [asdict(chunk) for chunk in self.chunks],
        }


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
        trace_steps: list[dict] = []

        # ── Direct hybrid retrieval ───────────────────────────────────
        chunks, direct_trace = self._retrieve_and_augment(query, self.top_k)
        direct_ok = self._sufficient(chunks)
        trace_steps.append(
            self._build_trace_step(
                key="direct",
                title="Direct Retrieval",
                strategy=RetrievalStrategy.DIRECT,
                status="selected" if direct_ok else "failed",
                query=query,
                reason=(
                    "Threshold ustu en az bir chunk bulundu."
                    if direct_ok
                    else "Yeterli grounded chunk bulunamadi; fallback zinciri baslatildi."
                ),
                chunks=chunks,
                substeps=direct_trace,
            )
        )
        if direct_ok:
            return RetrievalResult(
                chunks=chunks, strategy_used=RetrievalStrategy.DIRECT,
                queries_tried=queries_tried, grounded=True,
                threshold=self.threshold, trace_steps=trace_steps,
            )

        # ── Fallback 1: Query Expansion ──────────────────────────────
        logger.info("Direct retrieval insufficient → Query Expansion")
        variants = self.expander.expand(query)
        queries_tried.extend(variants[1:])

        expansion_chunks: list[RetrievedChunk] = list(chunks)
        for variant in variants[1:]:
            expansion_chunks.extend(
                self._retrieve_and_augment(variant, self.fallback_top_k)[0]
            )
        chunks = _deduplicate(expansion_chunks)[: self.fallback_top_k]
        _, expansion_trace = self._retrieve_and_augment(variants[-1], self.fallback_top_k)
        expansion_ok = self._sufficient(chunks)
        trace_steps.append(
            self._build_trace_step(
                key="expansion",
                title="Query Expansion",
                strategy=RetrievalStrategy.EXPANSION,
                status="selected" if expansion_ok else "failed",
                query=query,
                queries=variants,
                reason=(
                    "Varyant sorgular yeterli grounding sagladi."
                    if expansion_ok
                    else "Varyant sorgular denendi ancak threshold ustu yeterli sonuc cikmadi."
                ),
                chunks=chunks,
                substeps=expansion_trace,
            )
        )

        if expansion_ok:
            return RetrievalResult(
                chunks=chunks, strategy_used=RetrievalStrategy.EXPANSION,
                queries_tried=queries_tried, grounded=True,
                threshold=self.threshold, trace_steps=trace_steps,
            )

        # ── Fallback 2: HyDE ─────────────────────────────────────────
        logger.info("Expansion insufficient → HyDE")
        hypothesis = self.hyde.generate_hypothesis(query)
        queries_tried.append(f"[HyDE] {hypothesis[:80]}...")

        hyde_chunks, hyde_trace = self._retrieve_and_augment(hypothesis, self.fallback_top_k)
        chunks = _deduplicate(chunks + hyde_chunks)[: self.fallback_top_k]
        hyde_ok = self._sufficient(chunks)
        trace_steps.append(
            self._build_trace_step(
                key="hyde",
                title="HyDE",
                strategy=RetrievalStrategy.HYDE,
                status="selected" if hyde_ok else "failed",
                query=query,
                queries=[hypothesis],
                reason=(
                    "Hipotetik belge retrieval sonucunu guclendirdi."
                    if hyde_ok
                    else "Hipotetik belge denendi ancak yeterli grounding saglanamadi."
                ),
                chunks=chunks,
                substeps=hyde_trace,
            )
        )

        if hyde_ok:
            return RetrievalResult(
                chunks=chunks, strategy_used=RetrievalStrategy.HYDE,
                queries_tried=queries_tried, grounded=True,
                threshold=self.threshold, trace_steps=trace_steps,
            )

        # ── Fallback 3: Query Decomposition ──────────────────────────
        logger.info("HyDE insufficient → Query Decomposition")
        sub_questions = self.decomposer.decompose(query)
        queries_tried.extend(sub_questions)

        decomp_chunks: list[RetrievedChunk] = list(chunks)
        for sub_q in sub_questions:
            decomp_chunks.extend(self._retrieve_and_augment(sub_q, self.top_k)[0])
        chunks = _deduplicate(decomp_chunks)[: self.fallback_top_k]
        _, decomp_trace = self._retrieve_and_augment(sub_questions[-1], self.top_k) if sub_questions else ([], [])
        decomp_ok = self._sufficient(chunks)
        trace_steps.append(
            self._build_trace_step(
                key="decomposition",
                title="Query Decomposition",
                strategy=RetrievalStrategy.DECOMPOSITION,
                status="selected" if decomp_ok else "failed",
                query=query,
                queries=sub_questions,
                reason=(
                    "Alt sorular üzerinden yeterli chunk bulundu."
                    if decomp_ok
                    else "Alt sorular da yeterli grounding saglamadi."
                ),
                chunks=chunks,
                substeps=decomp_trace,
            )
        )

        if decomp_ok:
            return RetrievalResult(
                chunks=chunks,
                strategy_used=RetrievalStrategy.DECOMPOSITION,
                queries_tried=queries_tried,
                grounded=True,
                threshold=self.threshold,
                trace_steps=trace_steps,
            )

        trace_steps.append(
            {
                "key": "decision",
                "title": "Grounding Karari",
                "strategy": RetrievalStrategy.NONE,
                "status": "not_grounded",
                "query": query,
                "queries": [],
                "reason": "Tum asamalar denendi; threshold ustu yeterli kanit bulunamadi.",
                "threshold": self.threshold,
                "grounded_hits": 0,
                "chunks": [],
            }
        )
        return RetrievalResult(
            chunks=[], strategy_used=RetrievalStrategy.NONE,
            queries_tried=queries_tried, grounded=False,
            threshold=self.threshold, trace_steps=trace_steps,
        )

    # ------------------------------------------------------------------

    def _retrieve_and_augment(self, query: str, top_k: int) -> tuple[list[RetrievedChunk], list[dict]]:
        """Hybrid retrieval + relatives Q&A index augmentation."""
        # Main hybrid pipeline (dense + BM25 + rerank)
        hybrid_result = self.hybrid.retrieve_with_trace(query, top_k=top_k)
        raw_chunks = hybrid_result["ranked"]

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
        reranked = self._rerank(query, merged)
        return reranked, self._build_substeps(
            hybrid_result=hybrid_result,
            relatives_hits=relative_hits,
            parent_chunks=parent_chunks,
            merged_chunks=merged,
            reranked_chunks=reranked,
        )

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

    def _build_trace_step(
        self,
        *,
        key: str,
        title: str,
        strategy: RetrievalStrategy,
        status: str,
        chunks: list[RetrievedChunk],
        query: str = "",
        queries: list[str] | None = None,
        reason: str = "",
        substeps: list[dict] | None = None,
    ) -> dict:
        grounded_hits = len([chunk for chunk in chunks if chunk.score >= self.threshold])
        return RetrievalTraceStep(
            key=key,
            title=title,
            strategy=str(strategy),
            status=status,
            query=query,
            queries=list(queries or []),
            reason=reason,
            threshold=self.threshold,
            grounded_hits=grounded_hits,
            chunks=[
                RetrievalTraceChunk(
                    chunk_id=chunk.chunk_id,
                    document=chunk.doc_id or chunk.source,
                    page=chunk.page_num,
                    section=chunk.section,
                    score=round(chunk.score, 3),
                    content_type=str(chunk.content_type),
                    preview=(chunk.text or "")[:240],
                )
                for chunk in chunks[:8]
            ],
            substeps=list(substeps or []),
        ).as_dict()

    def _build_substeps(
        self,
        *,
        hybrid_result: dict,
        relatives_hits: list[RetrievedChunk],
        parent_chunks: list[RetrievedChunk],
        merged_chunks: list[RetrievedChunk],
        reranked_chunks: list[RetrievedChunk],
    ) -> list[dict]:
        return [
            self._build_substep("dense", "Dense Search", hybrid_result.get("dense", [])),
            self._build_substep("bm25", "BM25 Search", hybrid_result.get("bm25", [])),
            self._build_substep("rerank", "Cross-Encoder Rerank", hybrid_result.get("ranked", [])),
            self._build_substep("relatives", "Relatives Match", relatives_hits),
            self._build_substep("parents", "Parent Chunk Fetch", parent_chunks),
            self._build_substep("merge", "Merge + Dedup", merged_chunks),
            self._build_substep("lexical", "Lexical Re-rank", reranked_chunks),
        ]

    def _build_substep(
        self,
        key: str,
        title: str,
        chunks: list[RetrievedChunk],
    ) -> dict:
        return {
            "key": key,
            "title": title,
            "count": len(chunks),
            "chunks": [
                {
                    "chunk_id": chunk.chunk_id,
                    "document": chunk.doc_id or chunk.source,
                    "page": chunk.page_num,
                    "section": chunk.section,
                    "score": round(chunk.score, 3),
                    "content_type": str(chunk.content_type),
                    "preview": (chunk.text or "")[:180],
                }
                for chunk in chunks[:6]
            ],
        }
