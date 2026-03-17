"""
Vision extractor: uses a vision LLM (Claude) to extract structured content
from page images — tables, figures, charts — that plain text parsing misses.
The extracted content is stored alongside the raw text chunks.
"""
from __future__ import annotations

import anthropic

from src.config import get_settings
from src.ingestion.document_processor import DocumentPage


VISION_PROMPT = """\
You are an expert at analyzing document pages for an HR knowledge base.

Analyze this page image carefully and extract ALL content that may be missed
by plain text parsing:
- Tables (output as markdown table)
- Charts / graphs (describe key data points)
- Figures / diagrams (describe what they show)
- Formatted lists that lost structure
- Any other visually structured content

Return your extraction in this JSON format:
{
  "has_visual_content": true/false,
  "tables": [{"title": "...", "markdown": "..."}],
  "figures": [{"title": "...", "description": "..."}],
  "additional_text": "any text captured from visuals"
}

If the page has no visual content beyond plain text, set has_visual_content to false
and return empty arrays.
"""


class VisionExtractor:
    """Extracts structured content from page images using Claude vision."""

    def __init__(self):
        settings = get_settings()
        self.client = anthropic.Anthropic(api_key=settings.anthropic_api_key)
        self.model = settings.vision_model

    def extract(self, page: DocumentPage) -> dict:
        """
        Returns a dict with keys: has_visual_content, tables, figures, additional_text.
        Returns empty result if no image is available or extraction fails.
        """
        if not page.image_b64:
            return self._empty_result()

        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=2048,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                                "source": {
                                    "type": "base64",
                                    "media_type": "image/png",
                                    "data": page.image_b64,
                                },
                            },
                            {"type": "text", "text": VISION_PROMPT},
                        ],
                    }
                ],
            )

            import json
            text = response.content[0].text.strip()
            # Strip markdown code fences if present
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]
            return json.loads(text)

        except Exception as exc:
            # Non-fatal: log and return empty result
            print(f"[VisionExtractor] Warning: extraction failed for page "
                  f"{page.doc_id}:{page.page_num} — {exc}")
            return self._empty_result()

    def build_visual_text(self, extraction: dict) -> str:
        """Converts vision extraction dict into a text chunk for indexing."""
        parts: list[str] = []

        for table in extraction.get("tables", []):
            title = table.get("title", "Table")
            md = table.get("markdown", "")
            if md:
                parts.append(f"### {title}\n{md}")

        for fig in extraction.get("figures", []):
            title = fig.get("title", "Figure")
            desc = fig.get("description", "")
            if desc:
                parts.append(f"### {title}\n{desc}")

        additional = extraction.get("additional_text", "")
        if additional:
            parts.append(additional)

        return "\n\n".join(parts)

    @staticmethod
    def _empty_result() -> dict:
        return {
            "has_visual_content": False,
            "tables": [],
            "figures": [],
            "additional_text": "",
        }
