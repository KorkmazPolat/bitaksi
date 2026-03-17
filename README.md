# HR Assistant Chatbot

Çalışanların izin, yan haklar, sigorta ve İK prosedürleri gibi konularda doğal dilde soru sorup, **kaynak doküman ve sayfa referanslarıyla** güncel yanıt aldığı RAG tabanlı İK Bilgi Asistanı.

---

## Mimari

```
┌─────────────────────────────────────────────────────────────────────┐
│                        FastAPI Application                          │
│   POST /chat     POST /documents/ingest    POST /evaluation/*       │
└────────────┬────────────────────────────────────────────────────────┘
             │
      ┌──────▼──────────────────────────────────────┐
      │          Smart Grounding Retriever           │
      │                                              │
      │  Direct retrieval                            │
      │    ↓ (if < MIN_RESULTS above threshold)      │
      │  Fallback 1: Query Expansion                 │
      │    ↓ (still insufficient)                    │
      │  Fallback 2: HyDE                            │
      │    ↓ (still insufficient)                    │
      │  Fallback 3: Query Decomposition             │
      └──────┬──────────────────────┬───────────────┘
             │                      │
    ┌────────▼──────────┐  ┌────────▼──────────────┐
    │  raw_content index │  │  relatives index       │
    │  (ChromaDB)        │  │  (ChromaDB)            │
    │  Semantic chunks   │  │  Pre-generated Q&A     │
    │  + visual content  │  │  follow-up questions   │
    └────────────────────┘  └────────────────────────┘
             │
      ┌──────▼───────────────────────────┐
      │      Response Generator           │
      │  Claude LLM (RAG-mandatory)       │
      │  • Grounded answer only           │
      │  • Inline [Doc, p.N] citations    │
      │  • Sources section                │
      └───────────────────────────────────┘
```

### İki Paralel Index

| Index | İçerik | Amaç |
|-------|---------|-------|
| `hr_docs_raw` | Doküman bölüm chunk'ları + görsel içerik | Ana bilgi kaynağı |
| `hr_docs_relatives` | Her chunk için üretilen Q&A çiftleri | Soru-doküman eşleştirme |

---

## Belge Alma Pipeline'ı

```
PDF/DOCX
  → DocumentProcessor   (sayfa bazlı parse + tablo çıkarımı)
  → VisionExtractor     (Claude vision ile tablo/figür çıkarımı)
  → SemanticChunker     (bölüm başlığı farkındalıklı örtüşen chunk'lar)
  → DocumentIndexer     → raw_content index
                        → LLM ile related Q üretimi → relatives index
```

---

## Fallback Stratejileri

| Adım | Strateji | Açıklama |
|------|----------|----------|
| 0 | Direct | Ham sorgu ile vektör araması |
| 1 | Query Expansion | LLM ile anlamsal varyantlar üretimi |
| 2 | HyDE | Hipotetik doküman üretip gömme olarak kullanma |
| 3 | Query Decomposition | Karmaşık soruyu alt sorulara bölme |

---

## Değerlendirme Metrikleri

### Retrieval
- **Recall@K**: Üst K sonuçta bulunan ilgili chunk oranı
- **MRR**: Ortalama Karşılıklı Sıralama
- **NDCG@K**: Normalleştirilmiş İndirimli Kümülatif Kazanım

### Generation
- **Faithfulness**: Her iddianın bağlamla desteklenme oranı (LLM-as-judge)
- **Answer Relevancy**: Yanıtın soruyla ilgisi (1–5 → 0–1 normalize)

---

## Kurulum

```bash
cp .env.example .env
# .env içine ANTHROPIC_API_KEY ekle

pip install -r requirements.txt
uvicorn src.api.main:app --reload
```

### Docker

```bash
docker-compose up --build
```

---

## Kullanım

### Doküman Yükleme

```bash
curl -X POST http://localhost:8000/documents/ingest \
  -F "file=@hr_policy.pdf"
```

### Soru Sorma

```bash
curl -X POST http://localhost:8000/chat/ \
  -H "Content-Type: application/json" \
  -d '{"query": "Yıllık izin hakkım kaç gün?"}'
```

Yanıt örneği:
```json
{
  "answer": "Yıllık izin hakkınız **14 iş günü**dür. [İK El Kitabı, s.12]\n\nİzin taleplerini en az 2 hafta önceden HR sistemine girmeniz gerekmektedir.",
  "sources": [
    {"document": "ik_el_kitabi", "page": 12, "section": "İzin Politikası", "score": 0.923}
  ],
  "strategy_used": "direct",
  "grounded": true,
  "queries_tried": ["Yıllık izin hakkım kaç gün?"]
}
```

### Değerlendirme

```bash
curl -X POST http://localhost:8000/evaluation/retrieval \
  -H "Content-Type: application/json" \
  -d '{"retrieved_ids": ["c1","c2","c3"], "relevant_ids": ["c1","c2"], "k": 5}'
```

---

## Test

```bash
pytest tests/ -v
```

---

## Proje Yapısı

```
src/
├── api/
│   ├── main.py                  # FastAPI uygulaması
│   ├── dependencies.py          # DI container
│   └── routes/
│       ├── chat.py              # /chat endpoint
│       ├── documents.py         # /documents/ingest endpoint
│       ├── evaluation.py        # /evaluation/* endpoint'leri
│       └── health.py
├── ingestion/
│   ├── document_processor.py   # PDF/DOCX parse
│   ├── semantic_chunker.py     # Bölüm bazlı chunking
│   ├── vision_extractor.py     # Vision LLM ile tablo/figür çıkarımı
│   └── indexer.py              # Multi-index yöneticisi
├── retrieval/
│   ├── retriever.py            # Base ChromaDB retriever
│   ├── query_expansion.py      # Fallback 1
│   ├── hyde.py                 # Fallback 2
│   ├── query_decomposition.py  # Fallback 3
│   └── smart_grounding.py      # Orkestrasyon + dedup
├── generation/
│   └── response_generator.py  # RAG yanıt üretimi + alıntı
├── evaluation/
│   ├── retrieval_metrics.py    # Recall@K, MRR, NDCG
│   └── generation_metrics.py  # Faithfulness, Relevancy
└── config.py
```
