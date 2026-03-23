"""
Query Decomposition (Fallback Step 3):
Breaks a complex or multi-part question into atomic sub-questions,
retrieves for each, and merges results.
"""
from __future__ import annotations

import logging

from src.config import get_settings
from src.utils.llm import llm_call, parse_llm_json

logger = logging.getLogger(__name__)

DECOMP_PROMPT = """\
You are an HR knowledge assistant. The following employee question may contain
multiple sub-topics or be ambiguous. Break it down into 2-4 atomic, self-contained
sub-questions that together cover the original intent.

Original question: "{query}"

Return a JSON array of strings (sub-questions), no other text.
If the question is already atomic, return an array with just the original.
"""


class QueryDecomposer:
    """Decomposes complex queries into atomic sub-questions."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm_model

    def decompose(self, query: str) -> list[str]:
        """Returns a list of sub-questions (may be just [query] if simple)."""
        try:
            response = llm_call(
                model=self.model,
                max_tokens=400,
                messages=[
                    {"role": "user", "content": DECOMP_PROMPT.format(query=query)}
                ],
            )
            sub_questions = parse_llm_json(response)
            return sub_questions if sub_questions else [query]
        except Exception as exc:
            logger.warning("Query decomposition failed: %s", exc)
            return [query]
