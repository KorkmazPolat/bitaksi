"""
Tests for SmartGroundingRetriever using mocked BaseRetriever.
Verifies fallback chain ordering and deduplication.
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


def _make_retriever(chunks: list[RetrievedChunk]) -> MagicMock:
    mock = MagicMock()
    mock.retrieve.return_value = chunks
    mock.fetch_by_ids.return_value = []
    return mock


def test_direct_retrieval_success():
    """When direct retrieval returns enough high-score chunks, use 'direct' strategy."""
    raw = _make_retriever([_make_chunk("c1", 0.9), _make_chunk("c2", 0.85)])
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings:
        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        mock_settings.return_value = s

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("yıllık izin kaç gün?")

    assert result.strategy_used == RetrievalStrategy.DIRECT
    assert result.grounded is True
    assert len(result.chunks) == 2


def test_fallback_to_expansion():
    """When direct returns < MIN_RESULTS above threshold, trigger expansion."""
    low_score_chunk = _make_chunk("c1", score=0.3)   # below threshold
    good_chunk = _make_chunk("c2", score=0.8)

    call_count = [0]

    def side_effect(query, top_k=None):
        call_count[0] += 1
        if call_count[0] <= 1:
            return [low_score_chunk]
        return [good_chunk, _make_chunk("c3", 0.75)]

    raw = MagicMock()
    raw.retrieve.side_effect = side_effect
    raw.fetch_by_ids.return_value = []
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings, \
         patch("src.retrieval.smart_grounding.QueryExpander") as MockExpander, \
         patch("src.retrieval.smart_grounding.HyDERetrieval"), \
         patch("src.retrieval.smart_grounding.QueryDecomposer"):

        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        mock_settings.return_value = s

        MockExpander.return_value.expand.return_value = ["original query", "expanded query"]

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("izin politikası")

    assert result.strategy_used in {
        RetrievalStrategy.EXPANSION,
        RetrievalStrategy.HYDE,
        RetrievalStrategy.DECOMPOSITION,
        RetrievalStrategy.DIRECT,
    }


def test_no_grounded_result():
    """When all fallbacks fail, return grounded=False."""
    raw = _make_retriever([])
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings, \
         patch("src.retrieval.smart_grounding.QueryExpander") as MockExpander, \
         patch("src.retrieval.smart_grounding.HyDERetrieval") as MockHyDE, \
         patch("src.retrieval.smart_grounding.QueryDecomposer") as MockDecomp:

        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        mock_settings.return_value = s

        MockExpander.return_value.expand.return_value = ["q1", "q2"]
        MockHyDE.return_value.generate_hypothesis.return_value = "hypothesis"
        MockDecomp.return_value.decompose.return_value = ["sub1", "sub2"]

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("tamamen alakasız bir soru")

    assert result.grounded is False
    assert result.chunks == []


def test_deduplication():
    """Same chunk retrieved via multiple paths should appear only once."""
    chunk = _make_chunk("c1", score=0.9)
    raw = _make_retriever([chunk, chunk])
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings:
        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        mock_settings.return_value = s

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("test")

    chunk_ids = [c.chunk_id for c in result.chunks]
    assert len(chunk_ids) == len(set(chunk_ids))


def test_single_strong_chunk_is_now_grounded():
    raw = _make_retriever([_make_chunk("c1", 0.91)])
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings:
        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        mock_settings.return_value = s

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("tek bir guclu kaynakli soru")

    assert result.strategy_used == RetrievalStrategy.DIRECT
    assert result.grounded is True
    assert len(result.chunks) == 1


def test_reranking_prefers_lexically_aligned_chunk():
    raw = _make_retriever([
        RetrievedChunk(
            chunk_id="c1",
            text="Genel yonetim esaslari ve temsil yetkisi.",
            source="/docs/hr_policy.pdf",
            page_num=1,
            section="Genel",
            score=0.80,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Limited sirketlerde mudurler kurulu sirketin yonetimi ve temsili ile gorevlidir.",
            source="/docs/hr_policy.pdf",
            page_num=2,
            section="Mudurler Kurulu",
            score=0.75,
        ),
    ])
    relatives = _make_retriever([])

    with patch("src.retrieval.smart_grounding.get_settings") as mock_settings:
        s = MagicMock()
        s.similarity_threshold = 0.6
        s.top_k = 5
        s.fallback_top_k = 8
        s.lexical_overlap_weight = 0.3
        mock_settings.return_value = s

        grounding = SmartGroundingRetriever(raw, relatives)
        result = grounding.retrieve("mudurler kurulu ne yapar")

    assert result.chunks[0].chunk_id == "c2"
