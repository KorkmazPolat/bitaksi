"""
Query Expansion (Fallback Step 1):
Rewrites the original query into multiple semantically varied variants
to increase recall in the vector store.
"""
from __future__ import annotations

import json

import anthropic

from src.config import get_settings


EXPANSION_PROMPT = """\
You are an HR knowledge assistant. Rewrite the following employee question
into {n} semantically diverse variants that might retrieve relevant HR
policy documents. Include Turkish variants if the original is in Turkish.

Original question: "{query}"

Return a JSON array of {n} strings (the variants), no other text.
Example: ["variant 1", "variant 2", "variant 3"]
"""


class QueryExpander:
    """Generates multiple query variants to improve retrieval recall."""

    def __init__(self, n_variants: int = 3):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.llm_model
        self.n = n_variants

    def expand(self, query: str) -> list[str]:
        """Returns [original_query] + expanded variants (deduped)."""
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=512,
                messages=[
                    {
                        "role": "user",
                        "content": EXPANSION_PROMPT.format(
                            query=query, n=self.n
                        ),
                    }
                ],
            )
            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            variants = json.loads(text)
            # Deduplicate while preserving order; keep original first
            seen = {query}
            result = [query]
            for v in variants:
                if v not in seen:
                    seen.add(v)
                    result.append(v)
            return result
        except Exception as exc:
            print(f"[QueryExpander] Failed: {exc}")
            return [query]
