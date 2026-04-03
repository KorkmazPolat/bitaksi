"""
Base retriever: wraps ChromaDB collections for similarity search.
Returns RetrievedChunk objects with score, source, and page reference.
"""
from __future__ import annotations

from dataclasses import dataclass

from src.utils.enums import ContentType


@dataclass
class RetrievedChunk:
    chunk_id: str
    text: str
    source: str
    page_num: int
    section: str
    score: float           # cosine similarity (0–1, higher = better)
    doc_id: str = ""
    content_type: str = ContentType.TEXT

    @classmethod
    def from_metadata(
        cls,
        doc: str,
        meta: dict,
        score: float,
        chunk_id: str = "",
    ) -> "RetrievedChunk":
        """Construct from a raw ChromaDB document + metadata dict."""
        return cls(
            chunk_id=chunk_id or meta.get("chunk_id", ""),
            text=doc,
            source=meta.get("source", ""),
            page_num=int(meta.get("page_num", 0)),
            section=meta.get("section", ""),
            score=score,
            doc_id=meta.get("doc_id", ""),
            content_type=meta.get("content_type", ContentType.TEXT),
        )


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
            n_results=k,
            include=["documents", "metadatas", "distances"],
        )

        chunks: list[RetrievedChunk] = []
        ids = results.get("ids", [[]])[0]
        for cid, doc, meta, dist in zip(
            ids,
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ):
            meta = meta or {}
            score = 1.0 - dist   # cosine distance → similarity
            if score < self.score_threshold:
                continue
            # For relatives collection, metadata.chunk_id carries parent raw chunk id.
            resolved_chunk_id = str(meta.get("chunk_id") or cid)
            chunks.append(
                RetrievedChunk.from_metadata(
                    doc,
                    meta,
                    score,
                    chunk_id=resolved_chunk_id,
                )
            )

        return sorted(chunks, key=lambda c: c.score, reverse=True)

    def fetch_by_ids(
        self,
        chunk_ids: list[str],
        score_map: dict[str, float] | None = None,
        default_score: float = 0.0,
    ) -> list[RetrievedChunk]:
        """Fetch chunks by ID and attach caller-provided scores when available."""
        valid_ids = [cid for cid in chunk_ids if cid]
        if not valid_ids:
            return []
        results = self.collection.get(
            ids=valid_ids,
            include=["documents", "metadatas"],
        )
        chunks: list[RetrievedChunk] = []
        for cid, doc, meta in zip(
            results.get("ids", []),
            results.get("documents", []),
            results.get("metadatas", []),
        ):
            meta = meta or {}
            chunk_id = str(cid)
            score = (
                float(score_map.get(chunk_id, default_score))
                if score_map is not None
                else default_score
            )
            chunks.append(
                RetrievedChunk.from_metadata(doc, meta, score=score, chunk_id=chunk_id)
            )
        return chunks
