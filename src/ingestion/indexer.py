"""
Multi-index vector store manager.

Two ChromaDB collections:
  - raw_content   : semantic text chunks from document pages
  - relatives     : related/follow-up question pairs generated from each chunk

Vision extraction and related-question generation are parallelised with a
thread pool to avoid the sequential-API-call bottleneck on large documents.
"""
from __future__ import annotations

import hashlib
import logging
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import get_settings
from src.ingestion.document_processor import DocumentPage, DocumentProcessor
from src.ingestion.semantic_chunker import SemanticChunker, TextChunk
from src.ingestion.vision_extractor import VisionExtractor
from src.utils.enums import ContentType
from src.utils.llm import coerce_text_response, llm_call, parse_llm_json

logger = logging.getLogger(__name__)

RELATED_QUESTIONS_PROMPT = """\
Sen bir IK dokumani analiz asistanisin. Asagidaki dokuman parcasina bakarak
bir calisanin sorabilecegi TAM ve DOGAL sorular uret.

Kurallar:
- Ciktilar eksik ifade, baslik parcasi veya yarim cumle OLMAMALI.
- Her oge tek basina anlamli, tam bir soru olmali.
- Her soru mutlaka `?` ile bitmeli.
- Sorular dokumanin diliyle ayni dilde olmali.
- Sadece bu chunk'in cevaplayabilecegi sorular yaz.
- Cok genel veya anlamsiz sorular yazma.
- Her soruda mumkunse dokumandaki ana terimi koru.

Dokuman chunk:
\"\"\"
{chunk_text}
\"\"\"

Bolum: {section}
Kaynak: {source} (sayfa {page})

Yalnizca su formatta JSON don:
{{
  "questions": ["question 1", "question 2", ...],
  "follow_ups": ["follow-up 1", ...]
}}
"""

_MAX_WORKERS = 8   # thread pool size for parallel API calls
_QUESTION_WORD_RE = re.compile(
    r"\b(nasil|nedir|ne zaman|hangi|kim|kime|neden|niye|nerede|"
    r"nereden|ne kadar|kac|kaç|m[ıiuü])\b",
    re.IGNORECASE,
)
_INCOMPLETE_ENDINGS = (
    " ve",
    " veya",
    " ya da",
    " ile",
    " gibi",
    " icin",
    " için",
    " uzere",
    " üzere",
    " hakkinda",
    " hakkında",
)


class DocumentIndexer:
    """Ingests documents into the multi-index vector store."""

    def __init__(self):
        settings = get_settings()
        self.settings = settings

        embedding_fn = SentenceTransformerEmbeddingFunction(
            model_name=settings.embedding_model
        )

        Path(settings.chroma_persist_dir).mkdir(parents=True, exist_ok=True)
        self._client = chromadb.PersistentClient(path=settings.chroma_persist_dir)

        self.raw_collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_raw,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )
        self.relatives_collection = self._client.get_or_create_collection(
            name=settings.chroma_collection_relatives,
            embedding_function=embedding_fn,
            metadata={"hnsw:space": "cosine"},
        )

        self.processor = DocumentProcessor()
        self.chunker = SemanticChunker(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
        )
        self.vision = VisionExtractor()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> dict[str, Any]:
        """Full ingestion pipeline for a single file."""
        logger.info("Processing: %s", file_path)

        pages: list[DocumentPage] = self.processor.process(file_path)
        chunks: list[TextChunk] = self.chunker.chunk_pages(pages)

        visual_chunks = self._extract_visual_chunks_parallel(pages)
        all_chunks = chunks + visual_chunks

        logger.info(
            "%d chunks total (%d text + %d visual)",
            len(all_chunks), len(chunks), len(visual_chunks),
        )

        self._upsert_chunks(self.raw_collection, all_chunks)

        relatives = self._generate_relatives_parallel(all_chunks)
        if relatives:
            self._upsert_relatives(self.relatives_collection, relatives)

        return {
            "file": file_path,
            "pages": len(pages),
            "text_chunks": len(chunks),
            "visual_chunks": len(visual_chunks),
            "relatives": len(relatives),
        }

    def get_raw_collection(self):
        return self.raw_collection

    def get_relatives_collection(self):
        return self.relatives_collection

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_visual_chunks_parallel(
        self, pages: list[DocumentPage]
    ) -> list[TextChunk]:
        """Extract visual content from all pages in parallel."""
        visual_chunks: list[TextChunk] = []

        def _extract_one(page: DocumentPage) -> TextChunk | None:
            extraction = self.vision.extract(page)
            if not extraction.get("has_visual_content"):
                return None
            visual_text = self.vision.build_visual_text(extraction)
            if not visual_text.strip():
                return None
            return TextChunk(
                chunk_id=f"{page.doc_id}_p{page.page_num}_visual",
                doc_id=page.doc_id,
                source=page.source,
                page_num=page.page_num,
                section_title="Visual Content",
                text=visual_text,
                chunk_index=-1,
                metadata={
                    **page.metadata,
                    "content_type": ContentType.VISUAL,
                    "section": "Visual Content",
                },
            )

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_extract_one, page): page for page in pages}
            for future in as_completed(futures):
                try:
                    result = future.result()
                    if result:
                        visual_chunks.append(result)
                except Exception as exc:
                    page = futures[future]
                    logger.warning(
                        "Visual extraction failed for %s p%d: %s",
                        page.doc_id, page.page_num, exc,
                    )

        return visual_chunks

    def _generate_relatives_parallel(
        self, chunks: list[TextChunk]
    ) -> list[dict]:
        """Generate related questions for all eligible chunks in parallel."""
        eligible = [
            c for c in chunks
            if len(c.text.strip()) >= self.settings.min_chunk_length_for_relatives
        ]
        relatives: list[dict] = []

        def _gen_one(chunk: TextChunk) -> list[dict]:
            prompt = RELATED_QUESTIONS_PROMPT.format(
                chunk_text=chunk.text[: self.settings.chunk_preview_length],
                section=chunk.section_title,
                source=Path(chunk.source).name,
                page=chunk.page_num,
            )
            response = llm_call(
                model=self.settings.llm_model,
                max_tokens=512,
                messages=[{"role": "user", "content": prompt}],
            )
            data = parse_llm_json(coerce_text_response(response))
            results: list[dict] = []
            candidates = self._filter_relative_candidates(
                data.get("questions", []) + data.get("follow_ups", []),
                chunk=chunk,
            )
            for q in candidates:
                q_id = "q_" + hashlib.md5(
                    f"{chunk.chunk_id}:{q}".encode()
                ).hexdigest()
                results.append(
                    {
                        "id": q_id,
                        "question": q,
                        "chunk_id": chunk.chunk_id,
                        "source": chunk.source,
                        "page_num": chunk.page_num,
                        "section": chunk.section_title,
                    }
                )
            return results

        with ThreadPoolExecutor(max_workers=_MAX_WORKERS) as pool:
            futures = {pool.submit(_gen_one, chunk): chunk for chunk in eligible}
            for future in as_completed(futures):
                chunk = futures[future]
                try:
                    relatives.extend(future.result())
                except Exception as exc:
                    logger.warning(
                        "Relative generation failed for %s: %s", chunk.chunk_id, exc
                    )

        return relatives

    def _filter_relative_candidates(
        self,
        candidates: list[str],
        *,
        chunk: TextChunk,
    ) -> list[str]:
        """Drop low-signal LLM-generated questions before indexing."""
        seen: set[str] = set()
        filtered: list[str] = []
        min_len = self.settings.min_relative_question_length
        max_len = self.settings.max_relative_question_length
        anchor_terms = {
            token.casefold()
            for token in (
                chunk.section_title.split()
                + chunk.text.replace("\n", " ").split()
            )
            if len(token) >= 5
        }

        for raw in candidates:
            q = self._normalize_relative_question(str(raw or ""))
            if not q or q in seen:
                continue
            if len(q) < min_len or len(q) > max_len:
                continue
            if q.count("?") > 1:
                continue
            normalized = q.casefold()
            if normalized in {
                "bu nedir?",
                "detay nedir?",
                "bu konu nedir?",
            }:
                continue
            if len(q.split()) < 4:
                continue
            if not self._looks_like_complete_question(q):
                continue
            # Require some lexical grounding in the underlying chunk/section.
            q_terms = {tok.casefold().strip(".,:;!?()[]{}\"'") for tok in q.split()}
            q_terms = {tok for tok in q_terms if len(tok) >= 5}
            if anchor_terms and q_terms and not (q_terms & anchor_terms):
                continue
            seen.add(q)
            filtered.append(q)

        return filtered[:5]

    @staticmethod
    def _normalize_relative_question(raw: str) -> str:
        q = " ".join(str(raw or "").split())
        q = q.strip(" -•\t\"'")
        q = re.sub(r"^(soru|question|follow[- ]?up)\s*:\s*", "", q, flags=re.IGNORECASE)
        q = re.sub(r"\[[^\]]+\]", "", q)
        q = re.sub(r"\s+", " ", q).strip()
        q = q.rstrip(".,:;!-")
        q = re.sub(r"\?+$", "", q).strip()
        if not q:
            return ""
        q = q[0].upper() + q[1:] if len(q) > 1 else q.upper()
        return f"{q}?"

    @staticmethod
    def _looks_like_complete_question(q: str) -> bool:
        base = q[:-1].strip() if q.endswith("?") else q.strip()
        lower = base.casefold()
        if not base:
            return False
        if any(lower.endswith(ending) for ending in _INCOMPLETE_ENDINGS):
            return False
        if base.endswith(("(", "/", "-", ":", ",")):
            return False
        if not _QUESTION_WORD_RE.search(lower):
            if not re.search(r"\b\w+m[ıiuü]\b", lower):
                return False
        return True

    def _batch_upsert(
        self,
        collection,
        ids: list[str],
        documents: list[str],
        metadatas: list[dict],
    ) -> None:
        """Upsert in fixed-size batches to respect ChromaDB limits."""
        size = self.settings.batch_upsert_size
        for i in range(0, len(ids), size):
            collection.upsert(
                ids=ids[i: i + size],
                documents=documents[i: i + size],
                metadatas=metadatas[i: i + size],
            )

    def _upsert_chunks(self, collection, chunks: list[TextChunk]) -> None:
        if not chunks:
            return
        self._batch_upsert(
            collection,
            ids=[c.chunk_id for c in chunks],
            documents=[c.text for c in chunks],
            metadatas=[self._chunk_metadata(c) for c in chunks],
        )

    def _upsert_relatives(self, collection, relatives: list[dict]) -> None:
        if not relatives:
            return
        self._batch_upsert(
            collection,
            ids=[r["id"] for r in relatives],
            documents=[r["question"] for r in relatives],
            metadatas=[
                {
                    "chunk_id": r["chunk_id"],
                    "source": r["source"],
                    "page_num": r["page_num"],
                    "section": r["section"],
                }
                for r in relatives
            ],
        )

    @staticmethod
    def _chunk_metadata(c: TextChunk) -> dict:
        # parent_text stored in metadata so generation layer can retrieve
        # broader context without a second DB round-trip
        return {
            "doc_id": c.doc_id,
            "source": c.source,
            "page_num": c.page_num,
            "section": c.section_title,
            "breadcrumb": c.breadcrumb,
            "chunk_index": c.chunk_index,
            "token_count": str(c.token_count),
            "parent_text": c.parent_text[:2000],   # cap to avoid ChromaDB limits
            **{k: str(v) for k, v in c.metadata.items()
               if k not in ("section", "breadcrumb", "token_count")},
        }
