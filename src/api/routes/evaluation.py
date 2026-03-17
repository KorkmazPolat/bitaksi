"""
Evaluation routes: run retrieval and generation quality metrics.
"""
from __future__ import annotations

from fastapi import APIRouter
from pydantic import BaseModel

from src.evaluation.retrieval_metrics import RetrievalEvaluator
from src.evaluation.generation_metrics import GenerationEvaluator
from src.config import get_settings

router = APIRouter(prefix="/evaluation", tags=["evaluation"])


class RetrievalEvalRequest(BaseModel):
    retrieved_ids: list[str]
    relevant_ids: list[str]
    k: int = 5


class GenerationEvalRequest(BaseModel):
    question: str
    answer: str
    context_chunks: list[str]


class BatchRetrievalEvalRequest(BaseModel):
    dataset: list[dict]   # [{retrieved_ids, relevant_ids}]
    k: int = 5


class BatchGenerationEvalRequest(BaseModel):
    dataset: list[dict]   # [{question, answer, context_chunks}]


@router.post("/retrieval")
def evaluate_retrieval(request: RetrievalEvalRequest):
    """Compute Recall@K, MRR and NDCG@K for a single retrieval."""
    evaluator = RetrievalEvaluator()
    metrics = evaluator.evaluate(
        retrieved_ids=request.retrieved_ids,
        relevant_ids=set(request.relevant_ids),
        k=request.k,
    )
    return metrics.as_dict()


@router.post("/generation")
def evaluate_generation(request: GenerationEvalRequest):
    """Compute faithfulness and answer relevancy for a single answer."""
    evaluator = GenerationEvaluator()
    metrics = evaluator.evaluate(
        question=request.question,
        answer=request.answer,
        context_chunks=request.context_chunks,
    )
    return metrics.as_dict()


@router.post("/retrieval/batch")
def evaluate_retrieval_batch(request: BatchRetrievalEvalRequest):
    """Compute mean retrieval metrics over a dataset."""
    from src.evaluation.retrieval_metrics import BatchRetrievalEvaluator
    evaluator = BatchRetrievalEvaluator(k=request.k)
    return evaluator.evaluate_dataset(request.dataset)


@router.post("/generation/batch")
def evaluate_generation_batch(request: BatchGenerationEvalRequest):
    """Compute mean generation metrics over a dataset."""
    from src.evaluation.generation_metrics import BatchGenerationEvaluator
    evaluator = BatchGenerationEvaluator()
    return evaluator.evaluate_dataset(request.dataset)
