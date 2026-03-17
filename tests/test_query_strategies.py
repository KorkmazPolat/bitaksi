"""
Tests for query expansion, HyDE, and decomposition.
All Anthropic calls are routed through src.utils.llm.llm_call, which we mock.
"""
from unittest.mock import MagicMock, patch
import pytest


def _mock_message(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


# ── Query Expansion ──────────────────────────────────────────────────


def test_query_expansion_returns_list():
    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '["yıllık izin kaç gün", "annual leave days", "tatil hakkı"]'
    )):
        from src.retrieval.query_expansion import QueryExpander
        result = QueryExpander(n_variants=3).expand("yıllık izin hakkı nedir?")
        assert isinstance(result, list)
        assert result[0] == "yıllık izin hakkı nedir?"
        assert len(result) >= 1


def test_query_expansion_deduplicates():
    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '["yıllık izin hakkı nedir?", "yıllık izin hakkı nedir?"]'
    )):
        from src.retrieval.query_expansion import QueryExpander
        result = QueryExpander().expand("yıllık izin hakkı nedir?")
        assert result.count("yıllık izin hakkı nedir?") == 1


def test_query_expansion_fallback_on_error():
    with patch("src.utils.llm.llm_call", side_effect=Exception("API Error")):
        from src.retrieval.query_expansion import QueryExpander
        result = QueryExpander().expand("test query")
        assert result == ["test query"]


# ── HyDE ─────────────────────────────────────────────────────────────


def test_hyde_returns_hypothesis():
    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        "Employees are entitled to 14 days of annual leave per year."
    )):
        from src.retrieval.hyde import HyDERetrieval
        result = HyDERetrieval().generate_hypothesis("How many annual leave days?")
        assert len(result) > 10


def test_hyde_fallback_on_error():
    with patch("src.utils.llm.llm_call", side_effect=Exception("API Error")):
        from src.retrieval.hyde import HyDERetrieval
        query = "How many days of leave?"
        assert HyDERetrieval().generate_hypothesis(query) == query


# ── Query Decomposition ───────────────────────────────────────────────


def test_decomposition_splits_query():
    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '["Yıllık izin kaç gün?", "Hastalık izni prosedürü nedir?"]'
    )):
        from src.retrieval.query_decomposition import QueryDecomposer
        result = QueryDecomposer().decompose("Yıllık izin ve hastalık izni hakkında bilgi ver")
        assert isinstance(result, list)
        assert len(result) >= 1


def test_decomposition_fallback_on_error():
    with patch("src.utils.llm.llm_call", side_effect=Exception("API Error")):
        from src.retrieval.query_decomposition import QueryDecomposer
        query = "complex query"
        assert QueryDecomposer().decompose(query) == [query]
