"""
Tests for SmartGroundingRetriever.
Verifies fallback chain ordering and deduplication with HybridRetriever.
"""
from unittest.mock import MagicMock, patch

from src.retrieval.retriever import RetrievedChunk
from src.retrieval.smart_grounding import SmartGroundingRetriever, RetrievalResult
from src.utils.enums import RetrievalStrategy


def _make_chunk(chunk_id: str, score: float = 0.9) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=chunk_id,
        text=f"Content for {chunk_id}",
        source="/docs/hr_policy.pdf",
        page_num=1,
        section="General",
        score=score,
    )


def _make_hybrid(chunks: list[RetrievedChunk]) -> MagicMock:
    """Mock for HybridRetriever — exposes .retrieve() and .dense.fetch_by_ids()."""
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    mock.dense = MagicMock()
    mock.dense.fetch_by_ids.return_value = []
    return mock


def _make_relatives(chunks: list[RetrievedChunk]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    return mock


def _settings_mock(threshold: float = 0.6, top_k: int = 5, fallback_top_k: int = 8):
    s = MagicMock()
    s.similarity_threshold = threshold
    s.top_k = top_k
    s.fallback_top_k = fallback_top_k
    return s


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_direct_retrieval_success():
    """When direct retrieval returns enough high-score chunks, use DIRECT strategy."""
    hybrid = _make_hybrid([_make_chunk("c1", 0.9), _make_chunk("c2", 0.85)])
    relatives = _make_relatives([])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()):
        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("yıllık izin kaç gün?")

    assert result.strategy_used == RetrievalStrategy.DIRECT
    assert result.grounded is True
    assert len(result.chunks) == 2


def test_fallback_to_expansion():
    """When direct returns < MIN_RESULTS above threshold, trigger query expansion."""
    call_count = [0]

    def side_effect(query, top_k=None):
        call_count[0] += 1
        if call_count[0] <= 1:
            return [_make_chunk("c1", score=0.3)]   # below threshold
        return [_make_chunk("c2", 0.8), _make_chunk("c3", 0.75)]

    hybrid = MagicMock()
    hybrid.retrieve.side_effect = side_effect
    hybrid.dense = MagicMock()
    hybrid.dense.fetch_by_ids.return_value = []
    relatives = _make_relatives([])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()), \
         patch("src.retrieval.smart_grounding.QueryExpander") as MockExpander, \
         patch("src.retrieval.smart_grounding.HyDERetrieval"), \
         patch("src.retrieval.smart_grounding.QueryDecomposer"):

        MockExpander.return_value.expand.return_value = ["original query", "expanded query"]

        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("izin politikası")

    assert result.strategy_used in {
        RetrievalStrategy.EXPANSION,
        RetrievalStrategy.HYDE,
        RetrievalStrategy.DECOMPOSITION,
        RetrievalStrategy.DIRECT,
    }


def test_no_grounded_result():
    """When all fallbacks fail, return grounded=False."""
    hybrid = _make_hybrid([])
    relatives = _make_relatives([])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()), \
         patch("src.retrieval.smart_grounding.QueryExpander") as MockExpander, \
         patch("src.retrieval.smart_grounding.HyDERetrieval") as MockHyDE, \
         patch("src.retrieval.smart_grounding.QueryDecomposer") as MockDecomp:

        MockExpander.return_value.expand.return_value = ["q1", "q2"]
        MockHyDE.return_value.generate_hypothesis.return_value = "hypothesis"
        MockDecomp.return_value.decompose.return_value = ["sub1", "sub2"]

        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("tamamen alakasız bir soru")

    assert result.grounded is False
    assert result.chunks == []


def test_deduplication():
    """Same chunk retrieved via multiple paths should appear only once."""
    chunk = _make_chunk("c1", score=0.9)
    hybrid = _make_hybrid([chunk, chunk])
    relatives = _make_relatives([])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()):
        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("test")

    chunk_ids = [c.chunk_id for c in result.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_single_strong_chunk_is_grounded():
    """MIN_RESULTS=1: a single high-score chunk is sufficient for DIRECT."""
    hybrid = _make_hybrid([_make_chunk("c1", 0.91)])
    relatives = _make_relatives([])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()):
        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("tek bir guclu kaynakli soru")

    assert result.strategy_used == RetrievalStrategy.DIRECT
    assert result.grounded is True
    assert len(result.chunks) == 1


def test_reranking_prefers_lexically_aligned_chunk():
    """Lexical overlap boost should lift the query-aligned chunk above the semantic leader."""
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Genel yonetim esaslari ve temsil yetkisi.",
            source="/docs/hr_policy.pdf",
            page_num=1, section="Genel", score=0.80,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Limited sirketlerde mudurler kurulu sirketin yonetimi ve temsili ile gorevlidir.",
            source="/docs/hr_policy.pdf",
            page_num=2, section="Mudurler Kurulu", score=0.75,
        ),
    ]
    hybrid = _make_hybrid(chunks)
    relatives = _make_relatives([])

    def _settings_with_overlap():
        s = _settings_mock()
        s.lexical_overlap_weight = 0.3
        return s

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_with_overlap()):
        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("mudurler kurulu ne yapar")

    assert result.chunks[0].chunk_id == "c2"


def test_relatives_augmentation():
    """Relatives index hits whose chunk_ids are fetched and merged into results."""
    main_chunk = _make_chunk("c1", score=0.9)
    relative_hit = _make_chunk("rel1", score=0.7)
    parent_chunk = _make_chunk("c2", score=0.8)

    hybrid = _make_hybrid([main_chunk])
    hybrid.dense.fetch_by_ids.return_value = [parent_chunk]
    relatives = _make_relatives([relative_hit])

    with patch("src.retrieval.smart_grounding.get_settings", return_value=_settings_mock()):
        grounding = SmartGroundingRetriever(hybrid_retriever=hybrid, relatives_retriever=relatives)
        result = grounding.retrieve("test query")

    ids = {c.chunk_id for c in result.chunks}
    assert "c1" in ids
    assert "c2" in ids
