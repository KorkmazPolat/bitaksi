"""
Document ingestion routes: upload and index HR documents.
Also serves page images and text for the document viewer.
"""
from __future__ import annotations

import base64
import re
import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import get_settings
from src.ingestion.document_processor import PAGES_DIR
from src.ingestion.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}


class IngestResponse(BaseModel):
    file: str
    pages: int
    text_chunks: int
    visual_chunks: int
    relatives: int
    status: str


class PageViewerResponse(BaseModel):
    doc_id: str
    page_num: int
    image_b64: str | None          # PNG page render; None for DOCX virtual pages
    page_text: str                  # full page text
    highlighted_html: str           # page_text with <mark> around matched chunk
    has_image: bool


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index an HR document (PDF or DOCX)."""
    settings = get_settings()
    suffix = Path(file.filename or "").suffix.lower()

    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    max_bytes = settings.max_upload_size_mb * 1024 * 1024
    content = await file.read()
    if len(content) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=f"File exceeds maximum size of {settings.max_upload_size_mb} MB.",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(content)
        tmp_path = tmp.name

    try:
        indexer = DocumentIndexer()
        stats = indexer.ingest_file(tmp_path)
        return IngestResponse(**stats, status="success")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)


@router.get("/{doc_id}/page/{page_num}", response_model=PageViewerResponse)
def get_page(doc_id: str, page_num: int, chunk_text: str = ""):
    """
    Return a document page for the viewer panel.

    - image_b64 : PNG of the rendered PDF page (None for DOCX)
    - page_text  : full extracted text of the page
    - highlighted_html : page_text with the matching chunk wrapped in <mark>
    """
    # Load page image if it exists (PDF pages)
    img_path = PAGES_DIR / doc_id / f"page_{page_num}.png"
    image_b64: str | None = None
    if img_path.exists():
        image_b64 = base64.b64encode(img_path.read_bytes()).decode("utf-8")

    # Load page text from ChromaDB — query the raw collection for this exact page
    page_text = _fetch_page_text(doc_id, page_num)

    highlighted_html = _highlight_chunk(page_text, chunk_text) if chunk_text else _escape(page_text)

    return PageViewerResponse(
        doc_id=doc_id,
        page_num=page_num,
        image_b64=image_b64,
        page_text=page_text,
        highlighted_html=highlighted_html,
        has_image=image_b64 is not None,
    )


@router.get("/", response_model=list[dict])
def list_documents():
    """List all indexed documents (unique doc_ids) with page counts."""
    settings = get_settings()
    import chromadb
    from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(name=settings.chroma_collection_raw)
    except Exception:
        return []

    results = col.get(include=["metadatas"])
    docs: dict[str, dict] = {}
    for meta in results.get("metadatas", []):
        doc_id = meta.get("doc_id", "")
        page_num = int(meta.get("page_num", 0))
        if doc_id not in docs:
            docs[doc_id] = {"doc_id": doc_id, "max_page": 0, "has_images": False}
        docs[doc_id]["max_page"] = max(docs[doc_id]["max_page"], page_num)
        if (PAGES_DIR / doc_id).exists():
            docs[doc_id]["has_images"] = True

    return list(docs.values())


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_page_text(doc_id: str, page_num: int) -> str:
    """Pull the concatenated text for a given doc+page from ChromaDB."""
    settings = get_settings()
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(name=settings.chroma_collection_raw)
    except Exception:
        return ""

    results = col.get(
        where={"$and": [{"doc_id": doc_id}, {"page_num": page_num}]},
        include=["documents", "metadatas"],
    )

    docs = results.get("documents", [])
    if not docs:
        return ""

    # Join all chunks for this page in chunk_index order
    metas = results.get("metadatas", [])
    paired = sorted(
        zip(docs, metas),
        key=lambda x: int(x[1].get("chunk_index", 0)),
    )
    return "\n\n".join(d for d, _ in paired)


def _escape(text: str) -> str:
    """Minimal HTML escape for safe display."""
    return (
        text.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace("\n", "<br>")
    )


def _highlight_chunk(page_text: str, chunk_text: str) -> str:
    """
    Wrap the first occurrence of chunk_text inside page_text with <mark>.
    Falls back to a fuzzy match using the first 60 chars of the chunk.
    """
    escaped_page = _escape(page_text)
    escaped_chunk = _escape(chunk_text)

    # Try exact match first (on escaped strings so markup is consistent)
    if escaped_chunk and escaped_chunk in escaped_page:
        return escaped_page.replace(escaped_chunk, f'<mark class="hl">{escaped_chunk}</mark>', 1)

    # Fuzzy: try matching the first ~60 characters of the chunk
    snippet = _escape(chunk_text[:60].strip())
    if snippet and snippet in escaped_page:
        idx = escaped_page.index(snippet)
        end = min(idx + len(escaped_chunk), len(escaped_page))
        return (
            escaped_page[:idx]
            + f'<mark class="hl">{escaped_page[idx:end]}</mark>'
            + escaped_page[end:]
        )

    return escaped_page
