"""Tests for ResponseGenerator — mocks Anthropic API."""
from unittest.mock import MagicMock, patch
from src.retrieval.smart_grounding import RetrievalResult
from src.retrieval.retriever import RetrievedChunk


def _mock_response(text: str):
    msg = MagicMock()
    msg.content = [MagicMock(text=text)]
    return msg


def _make_chunk(chunk_id="c1", score=0.9):
    return RetrievedChunk(
        chunk_id=chunk_id,
        text="Yıllık izin 14 iş günüdür. [İK El Kitabı, s.12]",
        source="/docs/hr_policy.pdf",
        page_num=12,
        section="İzin Politikası",
        score=score,
    )


def test_generates_answer_with_sources():
    retrieval = RetrievalResult(
        chunks=[_make_chunk()],
        strategy_used="direct",
        grounded=True,
    )

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response(
            "Yıllık izin hakkınız 14 iş günüdür. [hr_policy, s.12]"
        )

        from src.generation.response_generator import ResponseGenerator
        gen = ResponseGenerator()
        gen.client = instance

        result = gen.generate(query="Yıllık iznim kaç gün?", retrieval_result=retrieval)

    assert result.grounded is True
    assert len(result.answer) > 0
    assert len(result.sources) >= 1
    assert result.sources[0]["page"] == 12


def test_no_context_returns_ungrounded():
    retrieval = RetrievalResult(
        chunks=[],
        strategy_used="none",
        grounded=False,
    )
    from src.generation.response_generator import ResponseGenerator, NO_CONTEXT_MSG
    gen = ResponseGenerator()
    result = gen.generate(query="anything", retrieval_result=retrieval)

    assert result.grounded is False
    assert result.answer == NO_CONTEXT_MSG
    assert result.sources == []


def test_source_deduplication():
    chunks = [
        _make_chunk("c1", score=0.9),
        _make_chunk("c1", score=0.9),  # duplicate
        _make_chunk("c2", score=0.8),
    ]
    # Use different pages to avoid dedup on page key
    chunks[2] = RetrievedChunk(
        chunk_id="c2", text="Different content", source="/docs/hr_policy.pdf",
        page_num=15, section="Sigorta", score=0.8,
    )

    retrieval = RetrievalResult(
        chunks=chunks, strategy_used="direct", grounded=True
    )

    with patch("anthropic.Anthropic") as MockClient:
        instance = MockClient.return_value
        instance.messages.create.return_value = _mock_response("answer")

        from src.generation.response_generator import ResponseGenerator
        gen = ResponseGenerator()
        gen.client = instance
        result = gen.generate("query", retrieval)

    pages = [s["page"] for s in result.sources]
    assert len(pages) == len(set(pages))   # no duplicate pages
