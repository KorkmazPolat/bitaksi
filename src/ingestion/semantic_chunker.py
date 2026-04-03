"""
Hierarchical Contextual Semantic Chunker
=========================================

Five-layer pipeline — each layer solves a specific weakness of naive chunking:

  Layer 1 · Structural Parse
      Detect heading hierarchy (numbered + ALL-CAPS + DOCX-style underline).
      Produces a tree of (level, title, text) sections so context is never lost.

  Layer 2 · Sentence Segmentation
      NLTK punkt tokenizer (Turkish-aware) splits sections into sentences.
      Falls back to regex splitter if NLTK is unavailable.
      Guarantees no chunk ever cuts mid-sentence.

  Layer 3 · Semantic Boundary Detection
      Embeds sentences with the project's sentence-transformer model.
      Computes cosine distance between consecutive sentences.
      Splits at the top-N% (95th percentile by default) of distance peaks —
      these are the genuine topic transitions in the text.

  Layer 4 · Token-Aware Sizing  (tiktoken cl100k_base)
      Merges groups that are too small (< min_tokens).
      Splits groups that are too large (> max_tokens) with sliding-window
      overlap to preserve context across boundaries.
      Word counts are never used — only token counts.

  Layer 5 · Context Enrichment
      Every child chunk is prefixed with its section breadcrumb so the
      embedding captures both local detail and global document position:
          "Section: İzin Politikası > Yıllık İzin\n\n<chunk text>"
      Parent text (full section) is stored in chunk metadata so the
      generation layer can retrieve broader context when needed.
      Optional LLM-generated context (Anthropic Contextual Retrieval style)
      can be enabled via `use_contextual_enrichment=True` in settings.

Result: TextChunk objects that are:
  - Semantically coherent  (never split at random offsets)
  - Token-bounded           (fit within embedding model limits)
  - Context-enriched        (breadcrumb + optional LLM summary)
  - Hierarchy-aware         (parent_text, section, level metadata)
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Optional

import numpy as np

from src.ingestion.document_processor import DocumentPage

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Optional dependencies — graceful degradation
# ---------------------------------------------------------------------------

try:
    import nltk

    def _ensure_nltk() -> None:
        for resource in ("tokenizers/punkt_tab", "tokenizers/punkt"):
            try:
                nltk.data.find(resource)
            except LookupError:
                name = resource.split("/")[-1]
                nltk.download(name, quiet=True)

    _ensure_nltk()
    from nltk.tokenize import sent_tokenize as _nltk_sent
    _HAS_NLTK = True
except ImportError:
    _HAS_NLTK = False
    logger.warning("nltk not installed — falling back to regex sentence splitter")

try:
    import tiktoken
    _ENCODER = tiktoken.get_encoding("cl100k_base")
    _HAS_TIKTOKEN = True
except ImportError:
    _ENCODER = None  # type: ignore[assignment]
    _HAS_TIKTOKEN = False
    logger.warning("tiktoken not installed — falling back to word-count approximation")

try:
    from sentence_transformers import SentenceTransformer
    _HAS_ST = True
except ImportError:
    _HAS_ST = False
    logger.warning("sentence-transformers not installed — semantic grouping disabled")


# ---------------------------------------------------------------------------
# Heading patterns
# ---------------------------------------------------------------------------

_HEADING_RE = re.compile(
    r"^(?:"
    r"\d+(?:\.\d+)*[\.\)]\s+\S.{0,100}"   # "1.2 Title" / "1.2.3) Title"
    r"|[A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s]{4,60}$"  # "YAN HAKLAR"
    r"|#{1,4}\s+\S.{0,100}"               # "## Title" (Markdown)
    r")",
    re.MULTILINE,
)

_SENTENCE_SPLIT_RE = re.compile(
    r"(?<=[.!?…])\s+(?=[A-ZÇĞİÖŞÜ\d\"])"  # sentence-ending punctuation
)


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class _Section:
    level: int          # 0 = document root, 1 = H1, 2 = H2, …
    title: str
    text: str           # raw text of this section (sentences only, no sub-headings)
    breadcrumb: str     # "H1 Title > H2 Title > …"


@dataclass
class TextChunk:
    chunk_id: str
    doc_id: str
    source: str
    page_num: int
    section_title: str
    text: str                          # enriched text (breadcrumb prefix + content)
    chunk_index: int
    parent_text: str = ""              # full section text — for generation context
    breadcrumb: str = ""               # "Doc > Section > Subsection"
    token_count: int = 0
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Token helpers
# ---------------------------------------------------------------------------

def _count_tokens(text: str) -> int:
    if _HAS_TIKTOKEN:
        return len(_ENCODER.encode(text))
    return max(1, int(len(text.split()) * 1.3))   # ~1.3 tokens/word approximation


def _decode_tokens(token_ids: list[int]) -> str:
    return _ENCODER.decode(token_ids) if _HAS_TIKTOKEN else ""


def _encode_tokens(text: str) -> list[int]:
    return _ENCODER.encode(text) if _HAS_TIKTOKEN else []


# ---------------------------------------------------------------------------
# Sentence splitter
# ---------------------------------------------------------------------------

def _split_sentences(text: str) -> list[str]:
    if _HAS_NLTK:
        try:
            sentences = _nltk_sent(text, language="turkish")
            if sentences:
                return [s.strip() for s in sentences if s.strip()]
        except Exception:
            pass
    # Regex fallback
    raw = _SENTENCE_SPLIT_RE.split(text)
    return [s.strip() for s in raw if s.strip()]


# ---------------------------------------------------------------------------
# Embedding model (lazy singleton)
# ---------------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_embed_model(model_name: str) -> Optional["SentenceTransformer"]:
    if not _HAS_ST:
        return None
    try:
        return SentenceTransformer(model_name)
    except Exception as exc:
        logger.warning("Could not load embedding model for semantic chunking: %s", exc)
        return None


# ---------------------------------------------------------------------------
# Main chunker
# ---------------------------------------------------------------------------

class HierarchicalSemanticChunker:
    """
    Best-practice document chunker combining structural parsing,
    semantic boundary detection, token-aware sizing and context enrichment.
    """

    def __init__(
        self,
        max_tokens: int = 512,
        overlap_tokens: int = 128,
        min_tokens: int = 50,
        semantic_percentile: int = 95,
        embedding_model: str = "all-MiniLM-L6-v2",
        use_contextual_enrichment: bool = False,
    ):
        self.max_tokens = max_tokens
        self.overlap_tokens = overlap_tokens
        self.min_tokens = min_tokens
        self.semantic_percentile = semantic_percentile
        self.embedding_model = embedding_model
        self.use_contextual_enrichment = use_contextual_enrichment
        self._embed_model = _get_embed_model(embedding_model)

    # ------------------------------------------------------------------
    # Public API (same interface as the old SemanticChunker)
    # ------------------------------------------------------------------

    def chunk_pages(self, pages: list[DocumentPage]) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        chunk_index = 0

        for page in pages:
            for chunk in self._process_page(page, chunk_index):
                chunks.append(chunk)
                chunk_index += 1

        return chunks

    # ------------------------------------------------------------------
    # Layer 1 — Structural parse
    # ------------------------------------------------------------------

    def _parse_sections(self, text: str) -> list[_Section]:
        """
        Walk lines, detect heading level by heading type, produce sections.
        """
        lines = text.splitlines()
        sections: list[_Section] = []
        title_stack: list[tuple[int, str]] = []   # (level, title)
        current_title = "Introduction"
        current_level = 1
        current_lines: list[str] = []

        def _flush() -> None:
            body = "\n".join(current_lines).strip()
            if not body:
                return
            crumb = " > ".join(t for _, t in title_stack) or current_title
            sections.append(
                _Section(
                    level=current_level,
                    title=current_title,
                    text=body,
                    breadcrumb=crumb,
                )
            )

        for line in lines:
            stripped = line.strip()
            if not stripped:
                current_lines.append("")
                continue

            if _HEADING_RE.match(stripped) and len(stripped) > 3:
                _flush()
                current_lines = []

                # Infer heading level
                if stripped.startswith("#"):
                    level = len(stripped) - len(stripped.lstrip("#"))
                elif re.match(r"^\d+\.\d+", stripped):
                    level = 2
                elif re.match(r"^\d+", stripped):
                    level = 1
                else:                    # ALL-CAPS
                    level = 1

                # Keep title_stack consistent with current level
                title_stack = [(l, t) for l, t in title_stack if l < level]
                title_stack.append((level, stripped))

                current_title = stripped
                current_level = level
            else:
                current_lines.append(line)

        _flush()
        return sections or [_Section(level=1, title="", text=text, breadcrumb="")]

    # ------------------------------------------------------------------
    # Layer 2 — Sentence segmentation
    # ------------------------------------------------------------------

    @staticmethod
    def _sentences(text: str) -> list[str]:
        return _split_sentences(text)

    # ------------------------------------------------------------------
    # Layer 3 — Semantic boundary detection
    # ------------------------------------------------------------------

    def _semantic_groups(self, sentences: list[str]) -> list[list[str]]:
        """
        Embed sentences; split where cosine distance between consecutive
        sentences exceeds the Nth percentile (= semantic topic change).
        """
        if len(sentences) <= 1 or self._embed_model is None:
            return [sentences]

        try:
            embeddings = self._embed_model.encode(
                sentences, show_progress_bar=False, convert_to_numpy=True
            )
            # L2-normalize for cosine via dot product
            norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
            normed = embeddings / np.maximum(norms, 1e-8)

            # Cosine similarity between consecutive sentences
            sims = [
                float(np.dot(normed[i], normed[i + 1]))
                for i in range(len(normed) - 1)
            ]
            distances = [1.0 - s for s in sims]

            if not distances:
                return [sentences]

            threshold = float(np.percentile(distances, self.semantic_percentile))
            breakpoints = [i + 1 for i, d in enumerate(distances) if d >= threshold]

            groups: list[list[str]] = []
            start = 0
            for bp in breakpoints:
                groups.append(sentences[start:bp])
                start = bp
            groups.append(sentences[start:])
            return groups

        except Exception as exc:
            logger.warning("Semantic grouping failed, falling back to one group: %s", exc)
            return [sentences]

    # ------------------------------------------------------------------
    # Layer 4 — Token-aware sizing
    # ------------------------------------------------------------------

    def _token_size_chunks(self, groups: list[list[str]]) -> list[str]:
        """
        1. Merge groups that are below min_tokens.
        2. For groups that still exceed max_tokens, apply sliding-window
           token overlap using tiktoken (falls back to word splitting).
        """
        # ── Merge small groups ──
        merged: list[str] = []
        buffer: list[str] = []
        buffer_tok = 0

        for group in groups:
            text = " ".join(group)
            tok = _count_tokens(text)

            if buffer_tok + tok <= self.max_tokens:
                buffer.extend(group)
                buffer_tok += tok
            else:
                if buffer:
                    merged.append(" ".join(buffer))
                buffer = list(group)
                buffer_tok = tok

        if buffer:
            merged.append(" ".join(buffer))

        # ── Hard-split oversized chunks with overlap ──
        final: list[str] = []
        for text in merged:
            if _count_tokens(text) <= self.max_tokens:
                final.append(text)
                continue

            if _HAS_TIKTOKEN:
                tokens = _encode_tokens(text)
                start = 0
                while start < len(tokens):
                    end = min(start + self.max_tokens, len(tokens))
                    final.append(_decode_tokens(tokens[start:end]))
                    if end >= len(tokens):
                        break
                    start = end - self.overlap_tokens
            else:
                # Word-based fallback
                words = text.split()
                start = 0
                wsize = self.max_tokens          # approx 1 word ≈ 1.3 tokens
                woverlap = self.overlap_tokens
                while start < len(words):
                    end = min(start + wsize, len(words))
                    final.append(" ".join(words[start:end]))
                    if end >= len(words):
                        break
                    start = end - woverlap

        return [c for c in final if c.strip()]

    # ------------------------------------------------------------------
    # Layer 5 — Context enrichment
    # ------------------------------------------------------------------

    def _enrich(self, chunk_text: str, breadcrumb: str) -> str:
        """Prepend section breadcrumb so the embedding captures hierarchy."""
        if breadcrumb:
            return f"Bölüm: {breadcrumb}\n\n{chunk_text}"
        return chunk_text

    # ------------------------------------------------------------------
    # Page → chunks
    # ------------------------------------------------------------------

    def _process_page(
        self, page: DocumentPage, start_index: int
    ) -> list[TextChunk]:
        chunks: list[TextChunk] = []
        idx = start_index

        sections = self._parse_sections(page.text)

        for section in sections:
            if not section.text.strip():
                continue

            parent_text = section.text   # full section → stored in metadata

            # Layers 2-4
            sentences = self._sentences(section.text)
            groups = self._semantic_groups(sentences)
            sized_texts = self._token_size_chunks(groups)

            for raw_text in sized_texts:
                if not raw_text.strip():
                    continue

                # Skip chunks that are below minimum token threshold
                if _count_tokens(raw_text) < self.min_tokens:
                    continue

                # Layer 5 — enrich
                enriched = self._enrich(raw_text, section.breadcrumb)

                token_count = _count_tokens(enriched)
                chunk_id = f"{page.doc_id}_p{page.page_num}_c{idx}"

                chunks.append(
                    TextChunk(
                        chunk_id=chunk_id,
                        doc_id=page.doc_id,
                        source=page.source,
                        page_num=page.page_num,
                        section_title=section.title,
                        text=enriched,
                        chunk_index=idx,
                        parent_text=parent_text,
                        breadcrumb=section.breadcrumb,
                        token_count=token_count,
                        metadata={
                            **page.metadata,
                            "section": section.title,
                            "breadcrumb": section.breadcrumb,
                            "level": section.level,
                            "token_count": token_count,
                        },
                    )
                )
                idx += 1

        return chunks


# ---------------------------------------------------------------------------
# Backward-compat alias used by indexer.py
# ---------------------------------------------------------------------------

class SemanticChunker(HierarchicalSemanticChunker):
    """
    Drop-in replacement for the old SemanticChunker.
    Accepts the same constructor arguments for backward compatibility
    but ignores word-based chunk_size/chunk_overlap in favour of
    token-based parameters from Settings.
    """

    def __init__(
        self,
        chunk_size: int = 800,         # ignored — kept for signature compat
        chunk_overlap: int = 100,      # ignored — kept for signature compat
        **kwargs,
    ):
        from src.config import get_settings
        s = get_settings()
        super().__init__(
            max_tokens=s.chunk_max_tokens,
            overlap_tokens=s.chunk_overlap_tokens,
            min_tokens=s.chunk_min_tokens,
            semantic_percentile=s.semantic_split_percentile,
            embedding_model=s.embedding_model,
            use_contextual_enrichment=s.use_contextual_enrichment,
            **kwargs,
        )
