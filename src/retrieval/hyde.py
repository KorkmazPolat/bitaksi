"""
HyDE – Hypothetical Document Embeddings (Fallback Step 2):
Generates a hypothetical HR policy answer for the query,
then retrieves using that synthetic document as the query embedding.
"""
from __future__ import annotations

import logging

from src.config import get_settings
from src.utils.llm import llm_call

logger = logging.getLogger(__name__)

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
        self.model = settings.llm_model

    def generate_hypothesis(self, query: str) -> str:
        """Returns a synthetic HR policy passage for the query."""
        try:
            response = llm_call(
                model=self.model,
                max_tokens=300,
                messages=[
                    {"role": "user", "content": HYDE_PROMPT.format(query=query)}
                ],
            )
            return response.strip()
        except Exception as exc:
            logger.warning("HyDE hypothesis generation failed: %s", exc)
            return query
