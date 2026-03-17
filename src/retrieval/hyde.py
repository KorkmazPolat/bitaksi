"""
HyDE – Hypothetical Document Embeddings (Fallback Step 2):
Generates a hypothetical HR policy answer for the query,
then retrieves using that synthetic document as the query embedding.
This bridges the gap between question-style queries and passage-style docs.
"""
from __future__ import annotations

import anthropic

from src.config import get_settings


HYDE_PROMPT = """\
You are writing a snippet from an HR policy document that directly answers
the following employee question. Write 3-5 sentences in a factual, policy-document
style (avoid hedging like "I think" or "maybe"). Be specific about rules, numbers,
and procedures if applicable. Write in the same language as the question.

Employee question: "{query}"

HR policy snippet:
"""


class HyDERetrieval:
    """Generates a hypothetical answer document for embedding-based retrieval."""

    def __init__(self):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

    def generate_hypothesis(self, query: str) -> str:
        """Returns a synthetic HR policy passage for the query."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=300,
                messages=[
                    {
                        "role": "user",
                        "content": HYDE_PROMPT.format(query=query),
                    }
                ],
            )
            return response.content[0].text.strip()
        except Exception as exc:
            print(f"[HyDE] Hypothesis generation failed: {exc}")
            return query  # fall back to original query
