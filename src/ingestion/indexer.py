"""
Multi-index vector store manager.

Two ChromaDB collections:
  - raw_content   : semantic text chunks from document pages
  - relatives     : related/follow-up question pairs generated from each chunk
                    (enables query-to-question matching for better recall)

Embeddings: sentence-transformers (all-MiniLM-L6-v2 by default).
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import chromadb
import anthropic
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

from src.config import get_settings
from src.ingestion.document_processor import DocumentPage, DocumentProcessor
from src.ingestion.semantic_chunker import SemanticChunker, TextChunk
from src.ingestion.vision_extractor import VisionExtractor


RELATED_QUESTIONS_PROMPT = """\
You are an HR knowledge assistant. Given the following HR document excerpt,
generate 3-5 natural language questions an employee might ask that this chunk
would answer. Also generate 1-2 follow-up questions.

Document chunk:
\"\"\"
{chunk_text}
\"\"\"

Section: {section}
Source: {source} (page {page})

Return a JSON object:
{{
  "questions": ["question 1", "question 2", ...],
  "follow_ups": ["follow-up 1", ...]
}}
"""


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
        self.llm = anthropic.Anthropic(api_key=settings.anthropic_api_key)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def ingest_file(self, file_path: str) -> dict[str, Any]:
        """Full ingestion pipeline for a single file."""
        print(f"[Indexer] Processing: {file_path}")

        pages: list[DocumentPage] = self.processor.process(file_path)
        chunks: list[TextChunk] = self.chunker.chunk_pages(pages)

        # Add visual content chunks from vision extraction
        visual_chunks = self._extract_visual_chunks(pages)
        all_chunks = chunks + visual_chunks

        print(f"[Indexer] {len(all_chunks)} chunks ({len(chunks)} text + "
              f"{len(visual_chunks)} visual)")

        # Index raw chunks
        self._upsert_chunks(self.raw_collection, all_chunks)

        # Generate and index related questions
        relatives = self._generate_relatives(all_chunks)
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

    def _extract_visual_chunks(self, pages: list[DocumentPage]) -> list[TextChunk]:
        visual_chunks: list[TextChunk] = []
        for page in pages:
            extraction = self.vision.extract(page)
            if not extraction.get("has_visual_content"):
                continue
            visual_text = self.vision.build_visual_text(extraction)
            if not visual_text.strip():
                continue
            chunk_id = f"{page.doc_id}_p{page.page_num}_visual"
            visual_chunks.append(
                TextChunk(
                    chunk_id=chunk_id,
                    doc_id=page.doc_id,
                    source=page.source,
                    page_num=page.page_num,
                    section_title="Visual Content",
                    text=visual_text,
                    chunk_index=-1,
                    metadata={
                        **page.metadata,
                        "content_type": "visual",
                        "section": "Visual Content",
                    },
                )
            )
        return visual_chunks

    def _upsert_chunks(self, collection, chunks: list[TextChunk]) -> None:
        if not chunks:
            return

        batch_size = 100
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i: i + batch_size]
            collection.upsert(
                ids=[c.chunk_id for c in batch],
                documents=[c.text for c in batch],
                metadatas=[
                    {
                        "doc_id": c.doc_id,
                        "source": c.source,
                        "page_num": c.page_num,
                        "section": c.section_title,
                        "chunk_index": c.chunk_index,
                        **{k: str(v) for k, v in c.metadata.items()},
                    }
                    for c in batch
                ],
            )

    def _generate_relatives(
        self, chunks: list[TextChunk]
    ) -> list[dict]:
        """Generate (question, chunk_id, metadata) triples via LLM."""
        relatives: list[dict] = []

        for chunk in chunks:
            if len(chunk.text.strip()) < 50:
                continue
            try:
                prompt = RELATED_QUESTIONS_PROMPT.format(
                    chunk_text=chunk.text[:1500],
                    section=chunk.section_title,
                    source=Path(chunk.source).name,
                    page=chunk.page_num,
                )
                response = self.llm.messages.create(
                    model=self.settings.llm_model,
                    max_tokens=512,
                    messages=[{"role": "user", "content": prompt}],
                )
                text = response.content[0].text.strip()
                if text.startswith("```"):
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                data = json.loads(text)

                for q in data.get("questions", []) + data.get("follow_ups", []):
                    q_id = "q_" + hashlib.md5(
                        f"{chunk.chunk_id}:{q}".encode()
                    ).hexdigest()
                    relatives.append(
                        {
                            "id": q_id,
                            "question": q,
                            "chunk_id": chunk.chunk_id,
                            "source": chunk.source,
                            "page_num": chunk.page_num,
                            "section": chunk.section_title,
                        }
                    )
            except Exception as exc:
                print(f"[Indexer] Relative generation failed for {chunk.chunk_id}: {exc}")

        return relatives

    def _upsert_relatives(
        self, collection, relatives: list[dict]
    ) -> None:
        if not relatives:
            return
        batch_size = 100
        for i in range(0, len(relatives), batch_size):
            batch = relatives[i: i + batch_size]
            collection.upsert(
                ids=[r["id"] for r in batch],
                documents=[r["question"] for r in batch],
                metadatas=[
                    {
                        "chunk_id": r["chunk_id"],
                        "source": r["source"],
                        "page_num": r["page_num"],
                        "section": r["section"],
                    }
                    for r in batch
                ],
            )
