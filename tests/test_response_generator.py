"""Tests for ResponseGenerator — patches the shared llm_call utility."""
from unittest.mock import MagicMock, patch
from src.retrieval.smart_grounding import RetrievalResult
from src.retrieval.retriever import RetrievedChunk
from src.utils.enums import RetrievalStrategy


def _mock_message(text: str):
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
        strategy_used=RetrievalStrategy.DIRECT,
        grounded=True,
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        "Yıllık izin hakkınız 14 iş günüdür. [hr_policy, s.12]"
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate(
            query="Yıllık iznim kaç gün?", retrieval_result=retrieval
        )

    assert result.grounded is True
    assert len(result.answer) > 0
    assert len(result.sources) >= 1
    assert result.sources[0]["page"] == 12


def test_no_context_returns_ungrounded():
    retrieval = RetrievalResult(
        chunks=[],
        strategy_used=RetrievalStrategy.NONE,
        grounded=False,
    )
    from src.generation.response_generator import ResponseGenerator, NO_CONTEXT_MSG
    result = ResponseGenerator().generate(query="anything", retrieval_result=retrieval)

    assert result.grounded is False
    assert result.answer == NO_CONTEXT_MSG
    assert result.sources == []
    assert "ilgili kaynak" in result.answer.lower()


def test_source_deduplication():
    chunks = [
        _make_chunk("c1", score=0.9),
        _make_chunk("c1", score=0.9),   # duplicate
        RetrievedChunk(
            chunk_id="c2", text="Different content",
            source="/docs/hr_policy.pdf", page_num=15,
            section="Sigorta", score=0.8,
        ),
    ]
    retrieval = RetrievalResult(
        chunks=chunks, strategy_used=RetrievalStrategy.DIRECT, grounded=True
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message("answer")):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    pages = [s["page"] for s in result.sources]
    assert len(pages) == len(set(pages))


def test_limits_sources_for_generation_context():
    chunks = [
        RetrievedChunk(
            chunk_id=f"c{i}",
            text=f"Chunk {i}",
            source="/docs/hr_policy.pdf",
            page_num=i,
            section=f"Section {i}",
            score=1.0 - (i * 0.01),
        )
        for i in range(1, 6)
    ]
    retrieval = RetrievalResult(
        chunks=chunks, strategy_used=RetrievalStrategy.DIRECT, grounded=True
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message("answer")):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    assert len(result.sources) <= 3
    assert result.sources[0]["page"] == 1
