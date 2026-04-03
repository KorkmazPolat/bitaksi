"""
HR Assistant Chatbot — FastAPI application entry point.
"""
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from src.api.routes import chat, documents, evaluation, health

app = FastAPI(
    title="HR Assistant Chatbot",
    description=(
        "Çalışanlara izin, yan haklar, sigorta ve İK prosedürleri hakkında "
        "kaynak doküman referanslarıyla yanıt veren RAG tabanlı asistan."
    ),
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve rendered PDF page images at /pages/{doc_id}/page_{n}.png
_pages_dir = Path("data/pages")
_pages_dir.mkdir(parents=True, exist_ok=True)
app.mount("/pages", StaticFiles(directory=str(_pages_dir)), name="pages")

# Serve the chat UI from src/api/static/
_static_dir = Path(__file__).parent / "static"
_static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(_static_dir)), name="static")


@app.get("/", include_in_schema=False)
def serve_chat_ui():
    """Serve the split-panel chat UI at the root URL."""
    return FileResponse(str(_static_dir / "index.html"))


app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(evaluation.router)
