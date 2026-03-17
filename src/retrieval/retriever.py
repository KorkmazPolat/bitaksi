"""
Base retriever: wraps ChromaDB collections for similarity search.
Returns RetrievedChunk objects with score, source, and page reference.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page_num: int
    section: str
    score: float           # cosine similarity (0–1, higher = better)
    doc_id: str = ""
    content_type: str = "text"   # "text" | "visual"


class BaseRetriever:
    """Retrieves chunks from a ChromaDB collection."""

    def __init__(self, collection, top_k: int = 5, score_threshold: float = 0.0):
        self.collection = collection
        self.top_k = top_k
        self.score_threshold = score_threshold

    def retrieve(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        k = top_k or self.top_k
        results = self.collection.query(
            query_texts=[query],
            n_results=min(k, self.collection.count() or 1),
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        for doc, meta, dist in zip(docs, metas, distances):
            # ChromaDB cosine distance → similarity
            score = 1.0 - dist
            if score < self.score_threshold:
                continue
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=meta.get("source", ""),
                    page_num=int(meta.get("page_num", 0)),
                    section=meta.get("section", ""),
                    score=score,
                    doc_id=meta.get("doc_id", ""),
                    content_type=meta.get("content_type", "text"),
                )
            )

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def retrieve_by_chunk_ids(
        self, chunk_ids: list[str]
    ) -> list[RetrievedChunk]:
        """Fetch chunks by their IDs (used after relatives lookup)."""
        if not chunk_ids:
            return []
        results = self.collection.get(
            ids=chunk_ids,
            include=["documents", "metadatas"],
        )
        chunks: list[RetrievedChunk] = []
        for doc, meta in zip(results["documents"], results["metadatas"]):
            chunks.append(
                RetrievedChunk(
                    chunk_id=meta.get("chunk_id", ""),
                    text=doc,
                    source=meta.get("source", ""),
                    page_num=int(meta.get("page_num", 0)),
                    section=meta.get("section", ""),
                    score=1.0,   # exact match
                    doc_id=meta.get("doc_id", ""),
                    content_type=meta.get("content_type", "text"),
                )
            )
        return chunks
