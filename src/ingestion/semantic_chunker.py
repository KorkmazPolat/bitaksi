"""
Semantic chunker: splits document pages into section-aware chunks.

Strategy:
1. Detect section boundaries via heading patterns.
2. Respect chunk_size / chunk_overlap limits.
3. Keep section title in each chunk's metadata for grounding.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

from src.ingestion.document_processor import DocumentPage


HEADING_PATTERN = re.compile(
    r"^(?:\d+[\.\d]*\s+[A-ZÇĞİÖŞÜa-zçğışöüA-Z].{0,80}|"  # numbered headings
    r"[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{4,50}$)",                    # ALL-CAPS headings
    re.MULTILINE,
)


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    source: str
    page_num: int
    section_title: str
    text: str
    chunk_index: int           # within the document
    metadata: dict = field(default_factory=dict)


class SemanticChunker:
    """Section-aware text chunker."""

    def __init__(self, chunk_size: int = 800, chunk_overlap: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def chunk_pages(self, pages: list[DocumentPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            sections = self._split_into_sections(page.text)
            for section_title, section_text in sections:
                for fragment in self._split_text(section_text):
                    chunk_id = f"{page.doc_id}_p{page.page_num}_c{chunk_index}"
                    chunks.append(
                        TextChunk(
                            chunk_id=chunk_id,
                            doc_id=page.doc_id,
                            source=page.source,
                            page_num=page.page_num,
                            section_title=section_title,
                            text=fragment,
                            chunk_index=chunk_index,
                            metadata={
                                **page.metadata,
                                "section": section_title,
                            },
                        )
                    )
                    chunk_index += 1
        return chunks

    # ------------------------------------------------------------------
    def _split_into_sections(self, text: str) -> list[tuple[str, str]]:
        """Return list of (section_title, section_text) pairs."""
        lines = text.splitlines()
        sections: list[tuple[str, str]] = []
        current_title = "Introduction"
        current_lines: list[str] = []

        for line in lines:
            if HEADING_PATTERN.match(line.strip()) and len(line.strip()) > 3:
                if current_lines:
                    sections.append((current_title, "\n".join(current_lines).strip()))
                current_title = line.strip()
                current_lines = []
            else:
                current_lines.append(line)

        if current_lines:
            sections.append((current_title, "\n".join(current_lines).strip()))

        return sections or [("", text)]

    def _split_text(self, text: str) -> list[str]:
        """Split text into overlapping fixed-size chunks."""
        if not text.strip():
            return []

        words = text.split()
        chunks: list[str] = []
        start = 0

        while start < len(words):
            end = start + self.chunk_size
            chunk = " ".join(words[start:end])
            chunks.append(chunk)
            if end >= len(words):
                break
            start = end - self.chunk_overlap

        return chunks
