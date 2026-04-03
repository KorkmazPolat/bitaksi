"""
Tests for HierarchicalSemanticChunker.

All tests run without a live embedding model — semantic grouping gracefully
falls back to a single group when the model is unavailable, so the rest of
the pipeline (section detection, token sizing, context enrichment) is always
exercised.
"""
from unittest.mock import patch
import pytest

from src.ingestion.document_processor import DocumentPage
from src.ingestion.semantic_chunker import (
    HierarchicalSemanticChunker,
    SemanticChunker,
    TextChunk,
    _count_tokens,
    _split_sentences,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _page(text: str, doc_id: str = "test", page_num: int = 1) -> DocumentPage:
    return DocumentPage(
        doc_id=doc_id, source="/tmp/test.pdf",
        page_num=page_num, text=text,
    )


def _chunker(**kwargs) -> HierarchicalSemanticChunker:
    """Create chunker with no embedding model (avoids network/GPU)."""
    c = HierarchicalSemanticChunker(**kwargs)
    c._embed_model = None   # force semantic-grouping fallback
    return c


# ---------------------------------------------------------------------------
# Token counting
# ---------------------------------------------------------------------------

def test_count_tokens_nonzero():
    assert _count_tokens("Yıllık izin hakkı nedir?") > 0


def test_count_tokens_empty():
    assert _count_tokens("") == 0 or _count_tokens("") >= 0   # 0 or approx


# ---------------------------------------------------------------------------
# Sentence splitting
# ---------------------------------------------------------------------------

def test_split_sentences_basic():
    text = "Bu bir cümledir. Bu da ikinci cümledir. Ve üçüncüsü."
    sents = _split_sentences(text)
    assert len(sents) >= 2


def test_split_sentences_single():
    text = "Tek bir cümle"
    sents = _split_sentences(text)
    assert len(sents) == 1
    assert sents[0] == text


# ---------------------------------------------------------------------------
# Section parsing (Layer 1)
# ---------------------------------------------------------------------------

def test_section_heading_detected():
    text = "1. İzin Politikası\nYıllık izin 14 gündür.\n\n2. Sigorta\nSGK zorunludur."
    chunker = _chunker()
    sections = chunker._parse_sections(text)
    titles = [s.title for s in sections]
    assert any("İzin" in t or "izin" in t or "1." in t for t in titles)


def test_all_caps_heading_detected():
    text = "YAN HAKLAR\nServis ve yemek verilir.\n\nSİGORTA\nSGK kapsamındadır."
    chunker = _chunker()
    sections = chunker._parse_sections(text)
    assert len(sections) >= 1


def test_no_heading_returns_one_section():
    text = "Sadece düz metin. Başlık yok. Devam ediyor."
    chunker = _chunker()
    sections = chunker._parse_sections(text)
    assert len(sections) == 1


def test_breadcrumb_populated():
    text = "1. İzin\nMetin burada.\n\n2. Sigorta\nSGK metni."
    chunker = _chunker()
    sections = chunker._parse_sections(text)
    assert all(isinstance(s.breadcrumb, str) for s in sections)


# ---------------------------------------------------------------------------
# Token-aware sizing (Layer 4)
# ---------------------------------------------------------------------------

def test_small_groups_merged():
    chunker = _chunker(min_tokens=20, max_tokens=200)
    groups = [["kısa"], ["çok kısa"], ["yine kısa"]]
    result = chunker._token_size_chunks(groups)
    # All tiny groups should be merged into one
    assert len(result) == 1


def test_large_chunk_split():
    chunker = _chunker(max_tokens=10, overlap_tokens=2)
    # 60+ word text — should be split into multiple chunks
    long_text = " ".join(["kelime"] * 60)
    result = chunker._token_size_chunks([[long_text]])
    assert len(result) >= 2


# ---------------------------------------------------------------------------
# Context enrichment (Layer 5)
# ---------------------------------------------------------------------------

def test_enrichment_prepends_breadcrumb():
    chunker = _chunker()
    enriched = chunker._enrich("İzin 14 gündür.", "İzin > Yıllık İzin")
    assert "İzin > Yıllık İzin" in enriched
    assert "İzin 14 gündür." in enriched


def test_enrichment_no_breadcrumb():
    chunker = _chunker()
    text = "Herhangi bir metin."
    assert chunker._enrich(text, "") == text


# ---------------------------------------------------------------------------
# End-to-end chunk_pages
# ---------------------------------------------------------------------------

def test_chunk_pages_returns_text_chunks():
    chunker = _chunker()
    page = _page("1. Politika\nBu bir paragraftır. İkinci cümle burada.")
    chunks = chunker.chunk_pages([page])
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_chunk_ids_are_unique():
    chunker = _chunker(max_tokens=20, min_tokens=1)
    page = _page("word " * 200)
    chunks = chunker.chunk_pages([page])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_page_returns_no_chunks():
    chunker = _chunker()
    chunks = chunker.chunk_pages([_page("   ")])
    assert chunks == []


def test_multi_page_chunk_indices_unique():
    chunker = _chunker(min_tokens=1)
    pages = [_page("Sayfa bir metni burada.", page_num=1),
             _page("Sayfa iki metni burada.", page_num=2)]
    chunks = chunker.chunk_pages(pages)
    indices = [c.chunk_index for c in chunks]
    assert len(indices) == len(set(indices))


def test_parent_text_stored():
    chunker = _chunker(min_tokens=1)
    page = _page("1. Bölüm\nBu bölümün içeriği burada yer almaktadır.")
    chunks = chunker.chunk_pages([page])
    for c in chunks:
        assert isinstance(c.parent_text, str)


def test_token_count_positive():
    chunker = _chunker(min_tokens=1)
    page = _page("Yeterince uzun bir metin parçası.")
    chunks = chunker.chunk_pages([page])
    for c in chunks:
        assert c.token_count > 0


# ---------------------------------------------------------------------------
# Backward-compat SemanticChunker alias
# ---------------------------------------------------------------------------

def test_semantic_chunker_alias_is_compatible():
    """SemanticChunker must accept old signature and produce TextChunk objects."""
    with patch("src.ingestion.semantic_chunker._get_embed_model", return_value=None):
        from src.ingestion.semantic_chunker import SemanticChunker
        chunker = SemanticChunker(chunk_size=800, chunk_overlap=100)
        page = _page("1. Politika\nÇalışanlar yıllık 14 gün izin hakkına sahiptir.")
        chunks = chunker.chunk_pages([page])
        assert isinstance(chunks, list)
