"""
Query Decomposition (Fallback Step 3):
Breaks a complex or multi-part question into atomic sub-questions,
retrieves for each, and merges results.
"""
from __future__ import annotations

import json

import anthropic

from src.config import get_settings


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
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model

    def decompose(self, query: str) -> list[str]:
        """Returns a list of sub-questions (may be just [query] if simple)."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=400,
                messages=[
                    {
                        "role": "user",
                        "content": DECOMP_PROMPT.format(query=query),
                    }
                ],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            sub_questions = json.loads(text)
            return sub_questions if sub_questions else [query]
        except Exception as exc:
            print(f"[QueryDecomposer] Failed: {exc}")
            return [query]
