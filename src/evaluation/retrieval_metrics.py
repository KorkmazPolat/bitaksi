"""
Retrieval evaluation metrics:
  - Recall@K   : fraction of relevant docs retrieved in top K
  - MRR        : Mean Reciprocal Rank
  - NDCG@K     : Normalized Discounted Cumulative Gain

Usage:
    from src.evaluation.retrieval_metrics import RetrievalEvaluator

    evaluator = RetrievalEvaluator()
    result = evaluator.evaluate(
        retrieved_ids=["chunk_1", "chunk_3", "chunk_7"],
        relevant_ids={"chunk_1", "chunk_3"},
        k=5,
    )
    print(result)
    # {"recall@5": 1.0, "mrr": 1.0, "ndcg@5": 1.0}
"""
from __future__ import annotations

import math
from dataclasses import dataclass


@dataclass
class RetrievalMetrics:
    recall_at_k: float
    mrr: float
    ndcg_at_k: float
    k: int

    def as_dict(self) -> dict:
        return {
            f"recall@{self.k}": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            f"ndcg@{self.k}": round(self.ndcg_at_k, 4),
        }


class RetrievalEvaluator:
    """Computes retrieval quality metrics against ground-truth relevance labels."""

    def evaluate(
        self,
        retrieved_ids: list[str],
        relevant_ids: set[str],
        k: int | None = None,
    ) -> RetrievalMetrics:
        """
        Args:
            retrieved_ids : Ordered list of retrieved chunk IDs (best first).
            relevant_ids  : Set of ground-truth relevant chunk IDs.
            k             : Cutoff. Defaults to len(retrieved_ids).
        """
        if k is None:
            k = len(retrieved_ids)

        top_k = retrieved_ids[:k]

        return RetrievalMetrics(
            recall_at_k=self._recall(top_k, relevant_ids),
            mrr=self._mrr(retrieved_ids, relevant_ids),
            ndcg_at_k=self._ndcg(top_k, relevant_ids),
            k=k,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _recall(top_k: list[str], relevant: set[str]) -> float:
        if not relevant:
            return 0.0
        hits = sum(1 for doc_id in top_k if doc_id in relevant)
        return hits / len(relevant)

    @staticmethod
    def _mrr(retrieved: list[str], relevant: set[str]) -> float:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0

    @staticmethod
    def _ndcg(top_k: list[str], relevant: set[str]) -> float:
        """Binary relevance NDCG."""
        def dcg(ids: list[str]) -> float:
            return sum(
                1.0 / math.log2(i + 2)
                for i, doc_id in enumerate(ids)
                if doc_id in relevant
            )

        actual_dcg = dcg(top_k)
        # Ideal: all relevant docs at top positions
        ideal_top_k = list(relevant)[: len(top_k)]
        ideal_dcg = dcg(ideal_top_k)
        return actual_dcg / ideal_dcg if ideal_dcg > 0 else 0.0


class BatchRetrievalEvaluator:
    """Evaluates retrieval over a dataset of (query, relevant_ids) pairs."""

    def __init__(self, k: int = 5):
        self.k = k
        self._evaluator = RetrievalEvaluator()

    def evaluate_dataset(
        self,
        dataset: list[dict],
    ) -> dict:
        """
        Args:
            dataset: list of {
                "query": str,
                "retrieved_ids": list[str],
                "relevant_ids": list[str],
            }
        Returns aggregated mean metrics.
        """
        recalls, mrrs, ndcgs = [], [], []

        for item in dataset:
            metrics = self._evaluator.evaluate(
                retrieved_ids=item["retrieved_ids"],
                relevant_ids=set(item["relevant_ids"]),
                k=self.k,
            )
            recalls.append(metrics.recall_at_k)
            mrrs.append(metrics.mrr)
            ndcgs.append(metrics.ndcg_at_k)

        n = len(dataset) or 1
        return {
            f"mean_recall@{self.k}": round(sum(recalls) / n, 4),
            "mean_mrr": round(sum(mrrs) / n, 4),
            f"mean_ndcg@{self.k}": round(sum(ndcgs) / n, 4),
            "num_queries": len(dataset),
        }
