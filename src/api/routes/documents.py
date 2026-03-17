"""
Document ingestion routes: upload and index HR documents.
"""
from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, HTTPException, UploadFile
from pydantic import BaseModel

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


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(file: UploadFile = File(...)):
    """Upload and index an HR document (PDF or DOCX)."""
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Allowed: {ALLOWED_EXTENSIONS}",
        )

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    try:
        indexer = DocumentIndexer()
        stats = indexer.ingest_file(tmp_path)
        return IngestResponse(**stats, status="success")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    finally:
        Path(tmp_path).unlink(missing_ok=True)
