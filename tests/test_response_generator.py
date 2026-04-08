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
        '{"answer_type":"answer","answer":"Yıllık izin hakkınız 14 iş günüdür. [hr_policy, s.12]"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate(
            query="Yıllık iznim kaç gün?", retrieval_result=retrieval
        )

    assert result.grounded is True
    assert result.answer_type == "answer"
    assert len(result.answer) > 0
    assert len(result.sources) >= 1
    assert len(result.citations) == 1
    assert result.sources[0]["page"] == 12
    assert result.sources[0]["highlight_text"] == "Yıllık izin 14 iş günüdür."
    assert result.citations[0]["source_index"] == 0
    assert result.citations[0]["evidence_text"] == "Yıllık izin 14 iş günüdür."


def test_no_context_returns_ungrounded():
    retrieval = RetrievalResult(
        chunks=[],
        strategy_used=RetrievalStrategy.NONE,
        grounded=False,
    )
    from src.generation.response_generator import ResponseGenerator, NO_CONTEXT_MSG
    result = ResponseGenerator().generate(query="anything", retrieval_result=retrieval)

    assert result.grounded is False
    assert result.answer_type == "abstain"
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

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"answer","answer":"answer"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    pages = [s["page"] for s in result.sources]
    assert len(pages) == len(set(pages))


def test_keeps_distinct_sources_on_same_page_when_chunks_differ():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Calisanlar yillik izin kullanabilir.",
            source="/docs/hr_policy.pdf",
            page_num=12,
            section="Izin Politikasi",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Izinler yonetici onayiyla planlanir.",
            source="/docs/hr_policy.pdf",
            page_num=12,
            section="Onay Sureci",
            score=0.85,
        ),
    ]
    retrieval = RetrievalResult(
        chunks=chunks, strategy_used=RetrievalStrategy.DIRECT, grounded=True
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"answer","answer":"answer"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    assert len(result.sources) == 2
    assert result.sources[0]["page"] == 12
    assert result.sources[1]["page"] == 12


def test_citations_map_to_distinct_sources_on_same_page():
    chunks = [
        RetrievedChunk(
            chunk_id="c1",
            text="Calisanlar yillik izin kullanabilir.",
            source="/docs/hr_policy.pdf",
            page_num=12,
            section="Izin Politikasi",
            score=0.9,
        ),
        RetrievedChunk(
            chunk_id="c2",
            text="Izinler yonetici onayiyla planlanir.",
            source="/docs/hr_policy.pdf",
            page_num=12,
            section="Onay Sureci",
            score=0.85,
        ),
    ]
    retrieval = RetrievalResult(
        chunks=chunks, strategy_used=RetrievalStrategy.DIRECT, grounded=True
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"answer","answer":"Calisanlar yillik izin kullanabilir [hr_policy, s.12]. Izinler yonetici onayiyla planlanir [hr_policy, s.12]."}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    assert len(result.citations) == 2
    assert result.citations[0]["source_index"] == 0
    assert result.citations[1]["source_index"] == 1
    assert result.citations[0]["evidence_text"] == "Calisanlar yillik izin kullanabilir."
    assert result.citations[1]["evidence_text"] == "Izinler yonetici onayiyla planlanir."


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

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"answer","answer":"answer"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        generator = ResponseGenerator()
        result = generator.generate("query", retrieval)

    assert len(result.sources) <= generator.max_sources_in_response
    assert result.sources[0]["page"] == 1


def test_highlight_text_prefers_best_matching_sentence():
    chunk = RetrievedChunk(
        chunk_id="c1",
        text=(
            "Yıllık izin 14 iş günüdür. "
            "İzinler birim yöneticisi onayıyla planlanır."
        ),
        source="/docs/hr_policy.pdf",
        page_num=12,
        section="İzin Politikası",
        score=0.9,
    )
    retrieval = RetrievalResult(
        chunks=[chunk],
        strategy_used=RetrievalStrategy.DIRECT,
        grounded=True,
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"answer","answer":"İzinler birim yöneticisi onayıyla planlanır. [hr_policy, s.12]"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("İzin nasıl planlanır?", retrieval)

    assert result.sources[0]["highlight_text"] == (
        "Yıllık izin 14 iş günüdür. İzinler birim yöneticisi onayıyla planlanır."
    )


def test_abstain_text_is_marked_as_abstain_even_with_retrieved_chunks():
    retrieval = RetrievalResult(
        chunks=[_make_chunk()],
        strategy_used=RetrievalStrategy.DIRECT,
        grounded=True,
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"abstain","answer":"Bu konuda dokumanlarda bilgi bulunmamaktadir. Netlestirmek icin IK ile iletisime geciniz."}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("query", retrieval)

    assert result.grounded is False
    assert result.answer_type == "abstain"
    assert result.citations == []


def test_clarify_response_is_preserved_without_sources():
    retrieval = RetrievalResult(
        chunks=[_make_chunk()],
        strategy_used=RetrievalStrategy.DIRECT,
        grounded=True,
    )

    with patch("src.utils.llm.llm_call", return_value=_mock_message(
        '{"answer_type":"clarify","answer":"Hangi izin turunu kastettiginizi netlestirebilir misiniz?"}'
    )):
        from src.generation.response_generator import ResponseGenerator
        result = ResponseGenerator().generate("buna hakkim var mi", retrieval)

    assert result.grounded is False
    assert result.answer_type == "clarify"
    assert result.sources == []
    assert result.citations == []
