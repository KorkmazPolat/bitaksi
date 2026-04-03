"""
Document ingestion routes: upload and index HR documents.
Also serves page images and text for the document viewer.
"""
from __future__ import annotations

import base64
import re
from pathlib import Path
from uuid import uuid4

from fastapi.responses import FileResponse

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.config import get_settings
from src.ingestion.document_processor import PAGES_DIR
from src.ingestion.indexer import DocumentIndexer

router = APIRouter(prefix="/documents", tags=["documents"])

ALLOWED_EXTENSIONS = {".pdf", ".docx", ".doc"}
UPLOADS_DIR = Path("data/uploads")


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


class ChunkDetail(BaseModel):
    id: str
    doc_id: str
    source: str
    page_num: int
    section: str
    chunk_index: int
    token_count: int
    breadcrumb: str
    content_type: str
    parent_text: str
    text: str
    metadata: dict


class ChunkListResponse(BaseModel):
    doc_id: str
    total_chunks: int
    chunks: list[ChunkDetail]


class RelativeQuestionDetail(BaseModel):
    id: str
    chunk_id: str
    source: str
    page_num: int
    section: str
    question: str
    metadata: dict


class RelativeQuestionListResponse(BaseModel):
    doc_id: str
    total_questions: int
    questions: list[RelativeQuestionDetail]


class DeleteDocumentResponse(BaseModel):
    doc_id: str
    deleted_raw_chunks: int
    deleted_relative_chunks: int
    deleted_files: int
    deleted_pages_dir: bool
    status: str


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

    safe_name = re.sub(r"[^a-zA-Z0-9_.-]", "_", Path(file.filename or "document").name)
    stem = Path(safe_name).stem or "document"
    ext = Path(safe_name).suffix or suffix
    unique_name = f"{stem}_{uuid4().hex[:8]}{ext}"
    UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    stored_path = UPLOADS_DIR / unique_name
    stored_path.write_bytes(content)

    try:
        indexer = DocumentIndexer()
        stats = indexer.ingest_file(str(stored_path))

        # Rebuild BM25 index so new document is searchable immediately
        try:
            from src.api.dependencies import get_chat_service
            get_chat_service().grounding.hybrid.invalidate_bm25()
        except Exception as exc:  # pragma: no cover
            import logging
            logging.getLogger(__name__).warning("BM25 rebuild skipped: %s", exc)

        return IngestResponse(**stats, status="success")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


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


@router.get("/{doc_id}/file")
def get_document_file(doc_id: str):
    """Serve original uploaded PDF file for in-browser rendering."""
    if not UPLOADS_DIR.exists():
        raise HTTPException(status_code=404, detail="Uploads directory not found.")

    matches = sorted(UPLOADS_DIR.glob(f"{doc_id}.pdf"))
    if not matches:
        raise HTTPException(status_code=404, detail="PDF file not found for this doc_id.")

    pdf_path = matches[0]
    return FileResponse(
        path=str(pdf_path),
        media_type="application/pdf",
        filename=pdf_path.name,
    )


@router.delete("/{doc_id}", response_model=DeleteDocumentResponse)
def delete_document(doc_id: str):
    """Delete a document and all related indexed/derived artifacts."""
    settings = get_settings()
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    raw_deleted = 0
    rel_deleted = 0
    files_deleted = 0
    pages_deleted = False

    # Raw chunks
    try:
        raw_col = client.get_collection(name=settings.chroma_collection_raw)
        raw_records = raw_col.get(where={"doc_id": doc_id}, include=[])
        raw_ids = raw_records.get("ids", [])
        if raw_ids:
            raw_col.delete(ids=raw_ids)
            raw_deleted = len(raw_ids)
    except Exception:
        pass

    # Relative chunks (linked via metadata.source path containing doc_id stem)
    try:
        rel_col = client.get_collection(name=settings.chroma_collection_relatives)
        rel_records = rel_col.get(include=["metadatas"])
        rel_ids: list[str] = []
        for rid, meta in zip(rel_records.get("ids", []), rel_records.get("metadatas", [])):
            source = str((meta or {}).get("source", ""))
            stem = Path(source).stem if source else ""
            if stem == doc_id:
                rel_ids.append(rid)
        if rel_ids:
            rel_col.delete(ids=rel_ids)
            rel_deleted = len(rel_ids)
    except Exception:
        pass

    # Uploaded originals
    if UPLOADS_DIR.exists():
        for p in UPLOADS_DIR.glob(f"{doc_id}.*"):
            try:
                p.unlink(missing_ok=True)
                files_deleted += 1
            except Exception:
                pass
        for p in UPLOADS_DIR.glob(f"{doc_id}_*.*"):
            try:
                p.unlink(missing_ok=True)
                files_deleted += 1
            except Exception:
                pass

    # Rendered page images
    pages_dir = PAGES_DIR / doc_id
    if pages_dir.exists() and pages_dir.is_dir():
        for child in pages_dir.glob("*"):
            child.unlink(missing_ok=True)
        pages_dir.rmdir()
        pages_deleted = True

    return DeleteDocumentResponse(
        doc_id=doc_id,
        deleted_raw_chunks=raw_deleted,
        deleted_relative_chunks=rel_deleted,
        deleted_files=files_deleted,
        deleted_pages_dir=pages_deleted,
        status="success",
    )


@router.get("/{doc_id}/chunks", response_model=ChunkListResponse)
def get_document_chunks(doc_id: str):
    """Return all raw chunks and metadata for a document."""
    settings = get_settings()
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(name=settings.chroma_collection_raw)
    except Exception:
        return ChunkListResponse(doc_id=doc_id, total_chunks=0, chunks=[])

    results = col.get(
        where={"doc_id": doc_id},
        include=["documents", "metadatas"],
    )

    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    rows = list(zip(ids, docs, metas))
    rows.sort(key=lambda row: (
        int((row[2] or {}).get("page_num", 0)),
        int((row[2] or {}).get("chunk_index", 0)),
    ))

    chunks: list[ChunkDetail] = []
    for chunk_id, text, meta in rows:
        m = meta or {}
        chunks.append(
            ChunkDetail(
                id=chunk_id,
                doc_id=str(m.get("doc_id", "")),
                source=str(m.get("source", "")),
                page_num=int(m.get("page_num", 0)),
                section=str(m.get("section", "")),
                chunk_index=int(m.get("chunk_index", 0)),
                token_count=int(m.get("token_count", 0)),
                breadcrumb=str(m.get("breadcrumb", "")),
                content_type=str(m.get("content_type", "text")),
                parent_text=str(m.get("parent_text", "")),
                text=text,
                metadata=m,
            )
        )

    return ChunkListResponse(
        doc_id=doc_id,
        total_chunks=len(chunks),
        chunks=chunks,
    )


@router.get("/{doc_id}/relatives", response_model=RelativeQuestionListResponse)
def get_document_relatives(doc_id: str):
    """Return generated relative questions for a document."""
    settings = get_settings()
    import chromadb

    client = chromadb.PersistentClient(path=settings.chroma_persist_dir)
    try:
        col = client.get_collection(name=settings.chroma_collection_relatives)
    except Exception:
        return RelativeQuestionListResponse(doc_id=doc_id, total_questions=0, questions=[])

    results = col.get(include=["documents", "metadatas"])
    ids = results.get("ids", [])
    docs = results.get("documents", [])
    metas = results.get("metadatas", [])

    questions: list[RelativeQuestionDetail] = []
    for rid, question, meta in zip(ids, docs, metas):
        m = meta or {}
        source = str(m.get("source", ""))
        stem = Path(source).stem if source else ""
        if stem != doc_id:
            continue
        questions.append(
            RelativeQuestionDetail(
                id=str(rid),
                chunk_id=str(m.get("chunk_id", "")),
                source=source,
                page_num=int(m.get("page_num", 0)),
                section=str(m.get("section", "")),
                question=str(question or ""),
                metadata=m,
            )
        )

    questions.sort(key=lambda item: (item.page_num, item.chunk_id, item.question))
    return RelativeQuestionListResponse(
        doc_id=doc_id,
        total_questions=len(questions),
        questions=questions,
    )


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


def _normalize_with_map(text: str) -> tuple[str, list[int]]:
    normalized_chars: list[str] = []
    index_map: list[int] = []
    previous_was_space = False

    for idx, ch in enumerate(text):
        lowered = ch.lower()
        if lowered.isalnum():
            normalized_chars.append(lowered)
            index_map.append(idx)
            previous_was_space = False
            continue

        if lowered.isspace() or lowered in "-_/.,:;!?()[]{}'\"":
            if normalized_chars and not previous_was_space:
                normalized_chars.append(" ")
                index_map.append(idx)
            previous_was_space = True
            continue

        if normalized_chars and not previous_was_space:
            normalized_chars.append(" ")
            index_map.append(idx)
        previous_was_space = True

    if normalized_chars and normalized_chars[-1] == " ":
        normalized_chars.pop()
        index_map.pop()

    return "".join(normalized_chars), index_map


def _highlight_chunk(page_text: str, chunk_text: str) -> str:
    """
    Wrap the first occurrence of chunk_text inside page_text with <mark>.
    Falls back to a fuzzy match using the first 60 chars of the chunk.
    """
    escaped_page = _escape(page_text)
    if not chunk_text.strip():
        return escaped_page

    normalized_page, page_map = _normalize_with_map(page_text)
    normalized_chunk, _ = _normalize_with_map(chunk_text)
    if not normalized_page or not normalized_chunk:
        return escaped_page

    idx = normalized_page.find(normalized_chunk)
    if idx == -1:
        snippet = normalized_chunk[: min(80, len(normalized_chunk))]
        idx = normalized_page.find(snippet) if snippet else -1
        if idx == -1:
            return escaped_page
        match_len = len(snippet)
    else:
        match_len = len(normalized_chunk)

    start = page_map[idx]
    end = page_map[min(idx + match_len - 1, len(page_map) - 1)] + 1
    return (
        _escape(page_text[:start])
        + f'<mark class="hl">{_escape(page_text[start:end])}</mark>'
        + _escape(page_text[end:])
    )
