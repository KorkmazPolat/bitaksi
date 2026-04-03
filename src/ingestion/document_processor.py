"""
Document processor: parses PDF and DOCX files into structured pages.
Each page carries its raw text plus base64-encoded image for vision LLM.
Page images are also persisted to data/pages/{doc_id}/page_{n}.png so
the document viewer can serve them without re-rendering.
"""
from __future__ import annotations

import base64
import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import pdfplumber
from docx import Document as DocxDocument

from src.config import get_settings

logger = logging.getLogger(__name__)

PAGES_DIR = Path("data/pages")


@dataclass
class DocumentPage:
    doc_id: str
    source: str          # file path
    page_num: int
    text: str
    image_b64: Optional[str] = None   # base64 PNG for vision extraction
    tables_raw: list[list[list[str]]] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)


class DocumentProcessor:
    """Parses PDF / DOCX files into DocumentPage objects."""

    SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".doc"}

    def process(self, file_path: str | Path) -> list[DocumentPage]:
        path = Path(file_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {file_path}")

        ext = path.suffix.lower()
        if ext == ".pdf":
            return self._process_pdf(path)
        elif ext in {".docx", ".doc"}:
            return self._process_docx(path)
        else:
            raise ValueError(f"Unsupported file type: {ext}")

    # ------------------------------------------------------------------
    # PDF
    # ------------------------------------------------------------------
    def _process_pdf(self, path: Path) -> list[DocumentPage]:
        pages: list[DocumentPage] = []
        doc_id = path.stem

        with pdfplumber.open(path) as pdf:
            for page_num, page in enumerate(pdf.pages, start=1):
                text = page.extract_text() or ""
                tables = page.extract_tables() or []

                # Render page to image — persist to disk AND keep in memory for vision
                save_path = PAGES_DIR / doc_id / f"page_{page_num}.png"
                image_b64 = self._render_pdf_page_to_b64(page, save_path=save_path)

                pages.append(
                    DocumentPage(
                        doc_id=doc_id,
                        source=str(path),
                        page_num=page_num,
                        text=text,
                        image_b64=image_b64,
                        tables_raw=tables,
                        metadata={"file_type": "pdf", "total_pages": len(pdf.pages)},
                    )
                )
        return pages

    @staticmethod
    def _render_pdf_page_to_b64(page, save_path: Optional[Path] = None) -> Optional[str]:
        try:
            img = page.to_image(resolution=150).original
            buf = io.BytesIO()
            img.save(buf, format="PNG")
            png_bytes = buf.getvalue()
            if save_path:
                save_path.parent.mkdir(parents=True, exist_ok=True)
                save_path.write_bytes(png_bytes)
            return base64.b64encode(png_bytes).decode("utf-8")
        except Exception as exc:
            logger.warning("PDF page render failed: %s", exc)
            return None

    # ------------------------------------------------------------------
    # DOCX
    # ------------------------------------------------------------------
    def _process_docx(self, path: Path) -> list[DocumentPage]:
        doc = DocxDocument(path)
        doc_id = path.stem
        pages: list[DocumentPage] = []

        # Group paragraphs into virtual "pages"
        paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
        page_size = get_settings().docx_page_paragraph_count
        for i in range(0, max(len(paragraphs), 1), page_size):
            chunk_paragraphs = paragraphs[i: i + page_size]
            text = "\n".join(chunk_paragraphs)
            page_num = i // page_size + 1

            pages.append(
                DocumentPage(
                    doc_id=doc_id,
                    source=str(path),
                    page_num=page_num,
                    text=text,
                    image_b64=None,
                    tables_raw=[],
                    metadata={"file_type": "docx"},
                )
            )
        return pages
