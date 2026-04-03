"""
Response generator: produces grounded, cited HR answers using Claude.

Key design decisions:
- RAG-mandatory: refuses to answer if no retrieved context is provided.
- Source citations: every answer includes document name + page references.
- Faithfulness enforcement: system prompt instructs Claude to stay within
  retrieved context (no hallucination).
- Supports multi-turn conversation via message history.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

from src.config import get_settings
from src.retrieval.smart_grounding import RetrievalResult
from src.utils.llm import llm_call

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Sen bir İK (İnsan Kaynakları) Bilgi Asistanısın. Görevin, kurum içi İK \
dokümanlarından gelen bağlam parçalarına dayanarak çalışanların sorularını \
yanıtlamaktır.

## Temel Kurallar
1. **YALNIZCA** sağlanan bağlam parçalarından elde ettiğin bilgilere dayanarak \
yanıt ver. Eğer yanıt bağlamda mevcut değilse, bunu açıkça belirt.
2. Her önemli ifadeyi, kaynak doküman adı ve sayfa numarasıyla alıntıla \
(örn. `[İK El Kitabı, s.12]`).
3. Çalışanın sorusunun dilinde yanıt ver (Türkçe/İngilizce).
4. Yanıtlar kısa, net ve uygulanabilir olmalıdır.
5. Emin olmadığında, çalışanı doğrudan İK departmanıyla iletişime geçmeye yönlendir.
6. Bağlamda tablo veya rakamlar varsa bunları yanıtına dahil et.

## Yanıt Formatı
- Başlık veya kısa özet
- Madde işaretli açıklama (gerekirse)
- Kaynaklar bölümü: doküman adı, bölüm ve sayfa numarası

## YASAK
- Bağlam dışından bilgi üretme (hallüsinasyon)
- "Bilmiyorum ama tahmin ediyorum ki…" gibi spekülasyonlar
- Kişisel görüş veya öneri (yalnızca politika aktarımı)
"""

NO_CONTEXT_MSG = """\
Üzgünüm, bu soruyu yanıtlamak için ilgili bir İK dokümanı bulunamadı. \
Lütfen sorunuzu farklı şekilde ifade etmeyi deneyin veya \
doğrudan **İK departmanıyla** iletişime geçin.
"""

_UNKNOWN_DOC = "Bilinmeyen Doküman"


@dataclass
class ChatMessage:
    role: Literal["user", "assistant"]
    content: str


@dataclass
class GenerationResult:
    answer: str
    sources: list[dict]          # [{document, page, section, score}]
    strategy_used: str
    grounded: bool
    queries_tried: list[str] = field(default_factory=list)


class ResponseGenerator:
    """Generates grounded HR answers with source citations."""

    def __init__(self):
        settings = get_settings()
        self.model = settings.llm_model
        self.max_history_turns = settings.max_history_turns

    def generate(
        self,
        query: str,
        retrieval_result: RetrievalResult,
        history: list[ChatMessage] | None = None,
    ) -> GenerationResult:

        if not retrieval_result.grounded or not retrieval_result.chunks:
            return GenerationResult(
                answer=NO_CONTEXT_MSG,
                sources=[],
                strategy_used=str(retrieval_result.strategy_used),
                grounded=False,
                queries_tried=retrieval_result.queries_tried,
            )

        context_block = self._format_context(retrieval_result.chunks)
        sources = self._extract_sources(retrieval_result.chunks)
        messages = self._build_messages(query, context_block, history or [])

        response = llm_call(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = response.strip()

        return GenerationResult(
            answer=answer,
            sources=sources,
            strategy_used=str(retrieval_result.strategy_used),
            grounded=True,
            queries_tried=retrieval_result.queries_tried,
        )

    # ------------------------------------------------------------------

    @staticmethod
    def _format_context(chunks) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc_name = Path(chunk.source).stem if chunk.source else _UNKNOWN_DOC
            parts.append(
                f"[Kaynak {i}: {doc_name} | Bölüm: {chunk.section} | Sayfa: {chunk.page_num}]\n"
                f"{chunk.text}"
            )
        return "\n\n---\n\n".join(parts)

    @staticmethod
    def _extract_sources(chunks) -> list[dict]:
        seen: set[str] = set()
        sources: list[dict] = []
        for chunk in chunks:
            doc_name = Path(chunk.source).stem if chunk.source else _UNKNOWN_DOC
            key = f"{doc_name}:{chunk.page_num}"
            if key not in seen:
                seen.add(key)
                sources.append(
                    {
                        "document": doc_name,
                        "page": chunk.page_num,
                        "section": chunk.section,
                        "score": round(chunk.score, 3),
                        "chunk_text": chunk.text,   # used by frontend for highlighting
                    }
                )
        return sorted(sources, key=lambda s: s["score"], reverse=True)

    def _build_messages(
        self,
        query: str,
        context: str,
        history: list[ChatMessage],
    ) -> list[dict]:
        messages: list[dict] = [
            {"role": m.role, "content": m.content}
            for m in history[-self.max_history_turns :]
        ]
        messages.append({
            "role": "user",
            "content": (
                f"## İlgili İK Dokümanı Bağlamı\n\n{context}\n\n"
                f"---\n\n## Çalışan Sorusu\n\n{query}"
            ),
        })
        return messages
