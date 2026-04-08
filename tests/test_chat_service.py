from unittest.mock import MagicMock

from src.api.dependencies import ChatService
from src.generation.response_generator import ChatMessage


def test_follow_up_query_is_augmented_for_retrieval():
    grounding = MagicMock()
    grounding.retrieve.return_value = MagicMock(chunks=[], trace_steps=[])
    generator = MagicMock()
    generator.generate.return_value = MagicMock(
        answer="Hangi izin turunu kastettiginizi netlestirebilir misiniz?",
        sources=[],
        citations=[],
        grounded=False,
        answer_type="clarify",
        strategy_used="direct",
        queries_tried=[],
    )
    debug_store = MagicMock()

    service = ChatService(grounding=grounding, generator=generator, debug_store=debug_store)
    history = [ChatMessage(role="user", content="Yillik izin kullanimi nasil ilerliyor?")]

    service.answer("Buna hakkim var mi?", history=history)

    grounding.retrieve.assert_called_once()
    retrieval_query = grounding.retrieve.call_args.args[0]
    assert "Yillik izin kullanimi nasil ilerliyor?" in retrieval_query
    assert "Takip sorusu: Buna hakkim var mi?" in retrieval_query


def test_standalone_query_is_not_augmented_for_retrieval():
    grounding = MagicMock()
    grounding.retrieve.return_value = MagicMock(chunks=[], trace_steps=[])
    generator = MagicMock()
    generator.generate.return_value = MagicMock(
        answer="Yanit",
        sources=[],
        citations=[],
        grounded=False,
        answer_type="abstain",
        strategy_used="direct",
        queries_tried=[],
    )
    debug_store = MagicMock()

    service = ChatService(grounding=grounding, generator=generator, debug_store=debug_store)
    history = [ChatMessage(role="user", content="Yillik izin kullanimi nasil ilerliyor?")]

    service.answer("Yillik izin talebi kac is gunu once acilmali?", history=history)

    retrieval_query = grounding.retrieve.call_args.args[0]
    assert retrieval_query == "Yillik izin talebi kac is gunu once acilmali?"
