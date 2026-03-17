"""
Tests for query expansion, HyDE, and decomposition.
These mock the Anthropic API so no real credentials are needed.
"""
from unittest.mock import MagicMock, patch
import pytest


def _mock_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ── Query Expansion ──────────────────────────────────────────────────


def test_query_expansion_returns_list():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response(
            '["yıllık izin kaç gün", "annual leave days", "tatil hakkı"]'
        )
        from src.retrieval.query_expansion import QueryExpander
        expander = QueryExpander(n_variants=3)
        expander.client = instance
        result = expander.expand("yıllık izin hakkı nedir?")
        assert isinstance(result, list)
        assert result[0] == "yıllık izin hakkı nedir?"   # original always first
        assert len(result) >= 1


def test_query_expansion_deduplicates():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response(
            '["yıllık izin hakkı nedir?", "yıllık izin hakkı nedir?"]'
        )
        from src.retrieval.query_expansion import QueryExpander
        expander = QueryExpander()
        expander.client = instance
        result = expander.expand("yıllık izin hakkı nedir?")
        assert result.count("yıllık izin hakkı nedir?") == 1


def test_query_expansion_fallback_on_error():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.side_effect = Exception("API Error")
        from src.retrieval.query_expansion import QueryExpander
        expander = QueryExpander()
        expander.client = instance
        result = expander.expand("test query")
        assert result == ["test query"]


# ── HyDE ─────────────────────────────────────────────────────────────


def test_hyde_returns_hypothesis():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response(
            "Employees are entitled to 14 days of annual leave per year."
        )
        from src.retrieval.hyde import HyDERetrieval
        hyde = HyDERetrieval()
        hyde.client = instance
        result = hyde.generate_hypothesis("How many annual leave days do I have?")
        assert "14" in result or len(result) > 10


def test_hyde_fallback_on_error():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.side_effect = Exception("API Error")
        from src.retrieval.hyde import HyDERetrieval
        hyde = HyDERetrieval()
        hyde.client = instance
        query = "How many days of leave?"
        result = hyde.generate_hypothesis(query)
        assert result == query   # falls back to original query


# ── Query Decomposition ───────────────────────────────────────────────


def test_decomposition_splits_query():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response(
            '["Yıllık izin kaç gün?", "Hastalık izni prosedürü nedir?"]'
        )
        from src.retrieval.query_decomposition import QueryDecomposer
        decomp = QueryDecomposer()
        decomp.client = instance
        result = decomp.decompose(
            "Yıllık izin ve hastalık izni hakkında bilgi ver"
        )
        assert isinstance(result, list)
        assert len(result) >= 1


def test_decomposition_fallback_on_error():
    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.side_effect = Exception("API Error")
        from src.retrieval.query_decomposition import QueryDecomposer
        decomp = QueryDecomposer()
        decomp.client = instance
        query = "complex query"
        result = decomp.decompose(query)
        assert result == [query]
