# Weekly Development Report — Week 4
**Developer:** Mert Kırtı
**Period:** March 15 – March 21, 2026
**Project:** HR Assistant Chatbot

---

## Summary

Final week focused on completing test coverage for the generation and strategy layers, full integration testing, and participating in the refactor pass.

---

## Completed Work

- **Unit Tests — Response Generator:** Wrote `tests/test_response_generator.py` covering: grounded answer generation with source citation, fallback refusal when `grounded=False`, multi-turn conversation history handling, and context formatting from `RetrievedChunk` objects.
- **Integration Testing:** Ran end-to-end tests with a real HR policy PDF through the full stack — upload, ingest, query with each retrieval strategy, evaluate retrieval and generation metrics. Verified source citations include correct document names, page numbers, and section titles.
- **Refactor Contribution:** Reviewed and cleaned up the ingestion pipeline during the final refactor pass — simplified parallelism in `DocumentIndexer`, tightened error handling in `VisionExtractor`, and removed redundant metadata fields.
- **pyproject.toml:** Set up `pyproject.toml` build configuration and `pytest` test runner settings.
- **Evaluation Validation:** Ran the batch evaluation endpoints against a small hand-labeled dataset to confirm Faithfulness and Answer Relevancy metrics behave correctly; verified NDCG calculation against a known reference implementation.

---

## Overall Contribution Highlights

- Document ingestion pipeline: `DocumentProcessor`, `SemanticChunker`, `VisionExtractor`, `DocumentIndexer`
- Dual ChromaDB index design (raw content + relatives/Q&A pairs)
- Full evaluation framework: retrieval metrics (Recall@K, MRR, NDCG) and generation metrics (Faithfulness, Answer Relevancy)
- Evaluation API endpoints
- Unit tests for semantic chunker, retrieval metrics, response generator
