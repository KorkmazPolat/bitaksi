"""
Unit tests for the production hybrid retrieval stack:
  - CrossEncoderReranker
  - BM25Retriever
  - HybridRetriever (RRF + rerank pipeline)
"""
from unittest.mock import MagicMock, patch
import numpy as np

import pytest

from src.retrieval.retriever import RetrievedChunk
from src.retrieval.reranker import CrossEncoderReranker
from src.retrieval.bm25_retriever import BM25Retriever, _tokenize
from src.retrieval.hybrid_retriever import HybridRetriever, _rrf_score, _rrf_fuse


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _chunk(chunk_id: str, text: str = "", score: float = 0.8) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=text or f"text for {chunk_id}",
        source="/docs/hr.pdf",
        page_num=1,
        section="HR",
        score=score,
    )


def _settings(
    use_bm25=True,
    use_reranking=True,
    rerank_model="cross-encoder/test",
    rerank_candidate_k=10,
    rerank_final_k=3,
    rrf_k=60,
):
    s = MagicMock()
    s.use_bm25 = use_bm25
    s.use_reranking = use_reranking
    s.rerank_model = rerank_model
    s.rerank_candidate_k = rerank_candidate_k
    s.rerank_final_k = rerank_final_k
    s.rrf_k = rrf_k
    return s


# ===========================================================================
# BM25Retriever
# ===========================================================================

class TestTokenize:
    def test_lowercases(self):
        tokens = _tokenize("Yıllık izin")
        assert tokens == ["yıllık", "izin"]

    def test_strips_punctuation(self):
        tokens = _tokenize("SGK Madde 4/a — prim")
        assert "sgk" in tokens
        assert "madde" in tokens
        assert "4" in tokens
        assert "a" in tokens
        assert "prim" in tokens

    def test_empty_string(self):
        assert _tokenize("") == []


class TestBM25Retriever:
    def _make_collection(self, docs, metas=None):
        col = MagicMock()
        metas = metas or [
            {
                "chunk_id": f"c{i}",
                "source": "/docs/hr.pdf",
                "page_num": 1,
                "section": "HR",
                "doc_id": "hr",
                "content_type": "text",
                "parent_text": "",
                "breadcrumb": "",
            }
            for i in range(len(docs))
        ]
        col.get.return_value = {"documents": docs, "metadatas": metas}
        return col

    def test_retrieve_returns_relevant_chunks(self):
        docs = [
            "yıllık izin hakkı çalışanlara tanınmıştır",
            "prim bildirimi SGK madde 4 kapsamında yapılır",
            "işten çıkma prosedürü insan kaynakları departmanınca yönetilir",
        ]
        col = self._make_collection(docs)

        with patch("src.retrieval.bm25_retriever.get_settings") as ms:
            ms.return_value.rerank_candidate_k = 10
            retriever = BM25Retriever(col)
            results = retriever.retrieve("yıllık izin")

        assert len(results) >= 1
        # Most relevant chunk (index 0) should rank first
        assert results[0].chunk_id == "c0"

    def test_retrieve_empty_collection(self):
        col = MagicMock()
        col.get.return_value = {"documents": [], "metadatas": []}

        with patch("src.retrieval.bm25_retriever.get_settings") as ms:
            ms.return_value.rerank_candidate_k = 10
            retriever = BM25Retriever(col)
            results = retriever.retrieve("herhangi bir şey")

        assert results == []

    def test_rebuild_clears_index(self):
        docs = ["izin belgesi gerekmektedir"]
        col = self._make_collection(docs)

        with patch("src.retrieval.bm25_retriever.get_settings") as ms:
            ms.return_value.rerank_candidate_k = 10
            retriever = BM25Retriever(col)
            retriever._ensure_index()
            assert retriever._bm25 is not None

            retriever.rebuild()
            # After rebuild, index is re-built from collection again
            assert retriever._bm25 is not None

    def test_graceful_degradation_on_missing_library(self):
        col = MagicMock()
        col.get.return_value = {"documents": ["test doc"], "metadatas": [{}]}

        with patch("src.retrieval.bm25_retriever.get_settings") as ms:
            ms.return_value.rerank_candidate_k = 10
            retriever = BM25Retriever(col)
            # Simulate rank_bm25 not installed
            with patch.dict("sys.modules", {"rank_bm25": None}):
                retriever._bm25 = None  # force rebuild
                retriever._chunks = []
                results = retriever.retrieve("test")
        # Should return empty list, not raise
        assert isinstance(results, list)


# ===========================================================================
# CrossEncoderReranker
# ===========================================================================

class TestCrossEncoderReranker:
    def test_rerank_sorts_by_score(self):
        chunks = [_chunk("c1", text="unrelated text"), _chunk("c2", text="yıllık izin politikası")]

        mock_model = MagicMock()
        # c2 should score higher for the izin query
        mock_model.predict.return_value = np.array([0.1, 0.9])

        with patch("src.retrieval.reranker.get_settings") as ms, \
             patch("src.retrieval.reranker._load_cross_encoder", return_value=mock_model):
            ms.return_value.rerank_model = "test-model"
            ms.return_value.rerank_final_k = 5
            reranker = CrossEncoderReranker()
            reranker._model = mock_model
            results = reranker.rerank("izin politikası", chunks, top_k=2)

        assert results[0].chunk_id == "c2"
        assert results[1].chunk_id == "c1"

    def test_rerank_respects_top_k(self):
        chunks = [_chunk(f"c{i}") for i in range(10)]
        scores = np.array([float(i) for i in range(10)])

        mock_model = MagicMock()
        mock_model.predict.return_value = scores

        with patch("src.retrieval.reranker.get_settings") as ms, \
             patch("src.retrieval.reranker._load_cross_encoder", return_value=mock_model):
            ms.return_value.rerank_model = "test-model"
            ms.return_value.rerank_final_k = 5
            reranker = CrossEncoderReranker()
            reranker._model = mock_model
            results = reranker.rerank("query", chunks, top_k=3)

        assert len(results) == 3

    def test_fallback_when_model_unavailable(self):
        chunks = [_chunk("c1", score=0.9), _chunk("c2", score=0.7)]

        with patch("src.retrieval.reranker.get_settings") as ms, \
             patch("src.retrieval.reranker._load_cross_encoder", return_value=None):
            ms.return_value.rerank_model = "test-model"
            ms.return_value.rerank_final_k = 5
            reranker = CrossEncoderReranker()
            results = reranker.rerank("query", chunks, top_k=2)

        # Falls back to original score order
        assert results[0].chunk_id == "c1"
        assert results[1].chunk_id == "c2"

    def test_empty_input(self):
        with patch("src.retrieval.reranker.get_settings") as ms, \
             patch("src.retrieval.reranker._load_cross_encoder", return_value=None):
            ms.return_value.rerank_model = "test-model"
            ms.return_value.rerank_final_k = 5
            reranker = CrossEncoderReranker()
            results = reranker.rerank("query", [])

        assert results == []


# ===========================================================================
# HybridRetriever — RRF fusion
# ===========================================================================

class TestRRF:
    def test_rrf_score_decreases_with_rank(self):
        k = 60
        assert _rrf_score(0, k) > _rrf_score(1, k) > _rrf_score(10, k)

    def test_rrf_fuse_combines_ranks(self):
        dense = [_chunk("c1", score=0.9), _chunk("c2", score=0.8)]
        bm25 = [_chunk("c2", score=0.7), _chunk("c3", score=0.6)]

        fused = _rrf_fuse(dense, bm25, rrf_k=60)
        ids = [c.chunk_id for c in fused]

        # c2 appeared in both lists → should have highest fused score
        assert ids[0] == "c2"
        assert set(ids) == {"c1", "c2", "c3"}

    def test_rrf_fuse_no_duplicates(self):
        chunk = _chunk("c1")
        fused = _rrf_fuse([chunk, chunk], [chunk], rrf_k=60)
        assert len(fused) == 1


class TestHybridRetriever:
    def _make_retriever(self, settings, dense_chunks, bm25_chunks=None, rerank_chunks=None):
        dense = MagicMock()
        dense.retrieve.return_value = dense_chunks

        bm25 = MagicMock()
        bm25.retrieve.return_value = bm25_chunks or []

        reranker = MagicMock()
        reranker.rerank.return_value = rerank_chunks or dense_chunks

        with patch("src.retrieval.hybrid_retriever.get_settings", return_value=settings):
            return HybridRetriever(
                dense_retriever=dense,
                bm25_retriever=bm25,
                reranker=reranker,
            ), dense, bm25, reranker

    def test_full_pipeline(self):
        s = _settings(use_bm25=True, use_reranking=True)
        chunks = [_chunk("c1"), _chunk("c2")]

        retriever, dense, bm25, reranker = self._make_retriever(
            s, dense_chunks=chunks, bm25_chunks=[], rerank_chunks=chunks[:1]
        )
        results = retriever.retrieve("izin politikası")

        dense.retrieve.assert_called_once()
        bm25.retrieve.assert_called_once()
        reranker.rerank.assert_called_once()
        assert len(results) == 1

    def test_bm25_disabled(self):
        s = _settings(use_bm25=False, use_reranking=False)
        chunks = [_chunk("c1"), _chunk("c2")]

        retriever, dense, bm25, reranker = self._make_retriever(s, dense_chunks=chunks)
        retriever.retrieve("query")

        bm25.retrieve.assert_not_called()

    def test_reranking_disabled(self):
        s = _settings(use_bm25=False, use_reranking=False)
        chunks = [_chunk("c1", score=0.9), _chunk("c2", score=0.7)]

        retriever, dense, bm25, reranker = self._make_retriever(s, dense_chunks=chunks)
        results = retriever.retrieve("query")

        reranker.rerank.assert_not_called()
        # Should still return sorted by score
        assert results[0].chunk_id == "c1"

    def test_empty_returns_empty(self):
        s = _settings(use_bm25=True, use_reranking=True)

        retriever, _, _, reranker = self._make_retriever(s, dense_chunks=[], bm25_chunks=[])
        results = retriever.retrieve("query")

        assert results == []
        reranker.rerank.assert_not_called()

    def test_invalidate_bm25_calls_rebuild(self):
        s = _settings()
        retriever, _, bm25, _ = self._make_retriever(s, dense_chunks=[])
        retriever.invalidate_bm25()
        bm25.rebuild.assert_called_once()
