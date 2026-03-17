"""
Generation quality metrics:
  - Faithfulness  : Is every claim in the answer supported by the context?
                    (LLM-as-judge approach — each sentence scored 0/1)
  - Answer Relevancy : How relevant is the answer to the original question?
                    (LLM scores 1–5, normalized to 0–1)

Both metrics use Claude as the evaluator (RAGAS-inspired).
"""
from __future__ import annotations

import json
from dataclasses import dataclass

import anthropic

from src.config import get_settings


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
    faithfulness_detail: list[dict]  # per-claim breakdown

    def as_dict(self) -> dict:
        return {
            "faithfulness": round(self.faithfulness, 4),
            "answer_relevancy": round(self.answer_relevancy, 4),
        }


class GenerationEvaluator:
    """LLM-as-judge evaluator for RAG answer quality."""

    def __init__(self):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

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
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                messages=[
                    {
                        "role": "user",
                        "content": FAITHFULNESS_PROMPT.format(
                            context=context[:4000],
                            answer=answer,
                        ),
                    }
                ],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            score = float(data.get("faithfulness_score", 0.0))
            claims = data.get("claims", [])
            return score, claims
        except Exception as exc:
            print(f"[GenerationEvaluator] Faithfulness eval failed: {exc}")
            return 0.0, []

    def _evaluate_relevancy(self, question: str, answer: str) -> float:
        try:
            response = self.client.messages.create(
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
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            data = json.loads(text)
            score = int(data.get("score", 3))
            return (score - 1) / 4.0   # normalize 1–5 → 0.0–1.0
        except Exception as exc:
            print(f"[GenerationEvaluator] Relevancy eval failed: {exc}")
            return 0.0


class BatchGenerationEvaluator:
    """Evaluates generation quality over a test dataset."""

    def __init__(self):
        self._evaluator = GenerationEvaluator()

    def evaluate_dataset(self, dataset: list[dict]) -> dict:
        """
        Args:
            dataset: list of {
                "question": str,
                "answer": str,
                "context_chunks": list[str],
            }
        """
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
