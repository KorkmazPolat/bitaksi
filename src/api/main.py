"""
HR Assistant Chatbot — FastAPI application entry point.
"""
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

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

app.include_router(health.router)
app.include_router(chat.router)
app.include_router(documents.router)
app.include_router(evaluation.router)
