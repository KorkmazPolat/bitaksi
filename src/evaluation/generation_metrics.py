"""
Generation quality metrics:
  - Faithfulness  : Is every claim in the answer supported by the context?
  - Answer Relevancy : How relevant is the answer to the original question?
"""
from __future__ import annotations

import logging
from dataclasses import dataclass

from src.config import get_settings
from src.utils.llm import llm_call, parse_llm_json

logger = logging.getLogger(__name__)

FAITHFULNESS_PROMPT = """\
You are evaluating whether an AI-generated answer is faithful to its source context.

## Context (retrieved HR documents):
{context}

## Generated Answer:
{answer}

## Task:
For each factual claim in the answer, determine if it is:
- Supported (1): directly supported by the context
- Unsupported (0): not found in the context or contradicts it

Return a JSON object:
{{
  "claims": [
    {{"claim": "claim text", "supported": 1 or 0, "reason": "brief reason"}}
  ],
  "faithfulness_score": <fraction of supported claims, 0.0–1.0>
}}
"""

RELEVANCY_PROMPT = """\
You are evaluating whether an AI-generated answer is relevant to the employee's question.

## Employee Question:
{question}

## Generated Answer:
{answer}

## Task:
Rate the relevancy of the answer to the question on a scale of 1–5:
  1 = Completely irrelevant
  2 = Mostly irrelevant, minor related content
  3 = Partially relevant
  4 = Mostly relevant, minor gaps
  5 = Fully relevant and complete

Return a JSON object:
{{
  "score": <1–5 integer>,
  "reason": "brief explanation"
}}
"""


@dataclass
class GenerationMetrics:
    faithfulness: float      # 0.0–1.0
    answer_relevancy: float  # 0.0–1.0
    faithfulness_detail: list[dict]

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
        }


class GenerationEvaluator:
    """LLM-as-judge evaluator for RAG answer quality."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm_model
        self.context_limit = settings.faithfulness_context_limit

    def evaluate(
        self,
        question: str,
        answer: str,
        context_chunks: list[str],
    ) -> GenerationMetrics:
        context = "\n\n---\n\n".join(context_chunks)
        faithfulness, detail = self._evaluate_faithfulness(answer, context)
        relevancy = self._evaluate_relevancy(question, answer)
        return GenerationMetrics(
            faithfulness=faithfulness,
            answer_relevancy=relevancy,
            faithfulness_detail=detail,
        )

    # ------------------------------------------------------------------

    def _evaluate_faithfulness(
        self, answer: str, context: str
    ) -> tuple[float, list[dict]]:
        try:
            response = llm_call(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": FAITHFULNESS_PROMPT.format(
                            context=context[: self.context_limit],
                            answer=answer,
                        ),
                    }
                ],
            )
            data = parse_llm_json(response.content[0].text)
            return float(data.get("faithfulness_score", 0.0)), data.get("claims", [])
        except Exception as exc:
            logger.warning("Faithfulness evaluation failed: %s", exc)
            return 0.0, []

    def _evaluate_relevancy(self, question: str, answer: str) -> float:
        try:
            response = llm_call(
                model=self.model,
                max_tokens=256,
                messages=[
                    {
                        "role": "user",
                        "content": RELEVANCY_PROMPT.format(
                            question=question, answer=answer
                        ),
                    }
                ],
            )
            data = parse_llm_json(response.content[0].text)
            score = int(data.get("score", 3))
            return (score - 1) / 4.0   # normalize 1–5 → 0.0–1.0
        except Exception as exc:
            logger.warning("Relevancy evaluation failed: %s", exc)
            return 0.0


class BatchGenerationEvaluator:
    """Evaluates generation quality over a test dataset."""

    def __init__(self):
        self._evaluator = GenerationEvaluator()

    def evaluate_dataset(self, dataset: list[dict]) -> dict:
        faithfulness_scores, relevancy_scores = [], []
        for item in dataset:
            metrics = self._evaluator.evaluate(
                question=item["question"],
                answer=item["answer"],
                context_chunks=item["context_chunks"],
            )
            faithfulness_scores.append(metrics.faithfulness)
            relevancy_scores.append(metrics.answer_relevancy)

        n = len(dataset) or 1
        return {
            "mean_faithfulness": round(sum(faithfulness_scores) / n, 4),
            "mean_answer_relevancy": round(sum(relevancy_scores) / n, 4),
            "num_samples": len(dataset),
        }
