"""Unit tests for retrieval metrics (no LLM calls)."""
import pytest
from src.evaluation.retrieval_metrics import RetrievalEvaluator, BatchRetrievalEvaluator


def test_perfect_retrieval():
    ev = RetrievalEvaluator()
    m = ev.evaluate(
        retrieved_ids=["a", "b", "c"],
        relevant_ids={"a", "b"},
        k=3,
    )
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    assert m.ndcg_at_k == pytest.approx(1.0, abs=0.01)


def test_no_relevant_retrieved():
    ev = RetrievalEvaluator()
    m = ev.evaluate(
        retrieved_ids=["x", "y", "z"],
        relevant_ids={"a", "b"},
        k=3,
    )
    assert m.recall_at_k == 0.0
    assert m.mrr == 0.0
    assert m.ndcg_at_k == 0.0


def test_mrr_second_position():
    ev = RetrievalEvaluator()
    m = ev.evaluate(
        retrieved_ids=["x", "a", "y"],
        relevant_ids={"a"},
        k=3,
    )
    assert m.mrr == pytest.approx(0.5)


def test_recall_partial():
    ev = RetrievalEvaluator()
    m = ev.evaluate(
        retrieved_ids=["a", "x", "y"],
        relevant_ids={"a", "b"},
        k=3,
    )
    assert m.recall_at_k == pytest.approx(0.5)


def test_batch_evaluator():
    ev = BatchRetrievalEvaluator(k=3)
    dataset = [
        {"retrieved_ids": ["a", "b"], "relevant_ids": ["a", "b"]},
        {"retrieved_ids": ["x", "y"], "relevant_ids": ["a", "b"]},
    ]
    result = ev.evaluate_dataset(dataset)
    assert result["mean_recall@3"] == pytest.approx(0.5)
    assert result["mean_mrr"] == pytest.approx(0.5)
    assert result["num_queries"] == 2


def test_ndcg_ordering():
    """Lower ranked relevant doc should have lower NDCG."""
    ev = RetrievalEvaluator()
    m_top = ev.evaluate(["a", "x", "y"], {"a"}, k=3)
    m_bottom = ev.evaluate(["x", "y", "a"], {"a"}, k=3)
    assert m_top.ndcg_at_k > m_bottom.ndcg_at_k
