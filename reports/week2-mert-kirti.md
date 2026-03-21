# Weekly Development Report — Week 2
**Developer:** Mert Kırtı
**Period:** March 1 – March 7, 2026
**Project:** HR Assistant Chatbot

---

## Summary

Completed the full ingestion pipeline. Vision-based extraction and the dual-index `DocumentIndexer` are now operational, enabling the retrieval layer to be connected.

---

## Completed Work

- **VisionExtractor:** Implemented `src/ingestion/vision_extractor.py`. Sends base64-encoded page images to the Claude Vision API and extracts structured content missed by text parsing — markdown tables, figure descriptions, and supplementary text. Returns JSON; handles failures gracefully with an empty-result fallback.
- **DocumentIndexer:** Built `src/ingestion/indexer.py` to manage two ChromaDB collections: `hr_docs_raw` (semantic text and visual chunks) and `hr_docs_relatives` (LLM-generated Q&A pairs per chunk). Parallelizes vision extraction and Q&A generation with 8 workers. Batch upserts chunks in groups of 100. Returns ingestion statistics (pages processed, text chunks, visual chunks, related questions).
- **Dual-Index Design:** Introduced the `relatives` index to generate question-answer pairs for each chunk, improving semantic match quality for employee queries that don't use exact policy language.
- **Integration Testing:** Ran a manual end-to-end test with a sample HR PDF — confirmed pages parsed, chunks indexed, and Q&A pairs generated correctly.

---

## Next Week

- Implement the evaluation metrics: Recall@K, MRR, NDCG for retrieval quality.
- Begin work on the `GenerationEvaluator` (faithfulness, answer relevancy scoring).
- Write unit tests for the semantic chunker and ingestion pipeline.
