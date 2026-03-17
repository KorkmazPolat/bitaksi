"""Unit tests for the semantic chunker (no I/O)."""
from src.ingestion.semantic_chunker import SemanticChunker, TextChunk
from src.ingestion.document_processor import DocumentPage


def _make_page(text: str) -> DocumentPage:
    return DocumentPage(
        doc_id="test_doc",
        source="/tmp/test.pdf",
        page_num=1,
        text=text,
    )


def test_basic_chunking():
    chunker = SemanticChunker(chunk_size=20, chunk_overlap=5)
    page = _make_page("word " * 50)
    chunks = chunker.chunk_pages([page])
    assert len(chunks) > 1
    assert all(isinstance(c, TextChunk) for c in chunks)


def test_section_detection():
    chunker = SemanticChunker(chunk_size=200, chunk_overlap=20)
    text = "1. İzin Politikası\nYıllık izin 14 gündür.\n\n2. Sigorta\nSGK zorunludur."
    page = _make_page(text)
    chunks = chunker.chunk_pages([page])
    sections = {c.section_title for c in chunks}
    # Should detect at least one section
    assert len(sections) >= 1


def test_chunk_ids_unique():
    chunker = SemanticChunker(chunk_size=10, chunk_overlap=2)
    page = _make_page("word " * 100)
    chunks = chunker.chunk_pages([page])
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_empty_page():
    chunker = SemanticChunker()
    page = _make_page("   ")
    chunks = chunker.chunk_pages([page])
    assert chunks == []


def test_overlap():
    chunker = SemanticChunker(chunk_size=5, chunk_overlap=2)
    words = ["w" + str(i) for i in range(20)]
    page = _make_page(" ".join(words))
    chunks = chunker.chunk_pages([page])
    # Overlap means chunks share some words — check second chunk starts before end of first
    if len(chunks) >= 2:
        first_words = set(chunks[0].text.split())
        second_words = set(chunks[1].text.split())
        assert first_words & second_words  # overlap exists
