# Weekly Development Report — Week 1
**Developer:** Mert Kırtı
**Period:** February 22 – February 28, 2026
**Project:** HR Assistant Chatbot

---

## Summary

Focused on the document ingestion pipeline. Built the PDF and DOCX parsing layer and the section-aware semantic chunker that will feed into the vector store.

---

## Completed Work

- **DocumentProcessor:** Implemented `src/ingestion/document_processor.py` to parse both PDF (`pdfplumber`) and DOCX (`python-docx`) files. Extracts page text, tables, and renders each page as a base64-encoded PNG image for later vision extraction. Returns structured `DocumentPage` objects.
- **SemanticChunker:** Built `src/ingestion/semantic_chunker.py` with heading-pattern detection for both Turkish and English section titles. Splits text into `TextChunk` objects with configurable `chunk_size` (default 800 chars) and `chunk_overlap` (100 chars). Preserves section metadata for grounding.
- **Dependency Research:** Evaluated embedding model options and selected `all-MiniLM-L6-v2` from `sentence-transformers` as the local embedding model. Validated ChromaDB 0.5.23 as the vector store.
- **requirements.txt Bootstrap:** Collected and pinned all ingestion-layer dependencies: `pdfplumber`, `python-docx`, `Pillow`, `sentence-transformers`, `chromadb`.

---

## Next Week

- Implement the `VisionExtractor` to analyze page images with Claude Vision API.
- Build the `DocumentIndexer` to manage dual ChromaDB indices and batch upserts.
- Align with Polat on the data contracts between `DocumentIndexer` output and `BaseRetriever` input.
