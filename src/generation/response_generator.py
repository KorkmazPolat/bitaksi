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
import re
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
7. Cevabı **doğrudan ver**; yalnızca "X için [doküman, s.Y]'ye bakınız" tarzı yönlendirme cümlesi yazma.
8. Gereksiz citation tekrarı yapma. Aynı paragrafta aynı kaynağı tekrar tekrar yazma.
9. Mümkünse cevabı 1-2 kısa paragraf veya en fazla 3 madde ile ver.
10. Toplam citation sayısını düşük tut; genelde 1-3 citation yeterlidir.

## Yanıt Formatı
- Kısa bir doğrudan cevap
- Gerekirse madde işaretleri
- Gerekli yerlerde kaynak citation
- Eğer bilgi eksikse, eksik olan kısmı açıkça belirt

## YASAK
- Bağlam dışından bilgi üretme (hallüsinasyon)
- "Bilmiyorum ama tahmin ediyorum ki…" gibi spekülasyonlar
- Kişisel görüş veya öneri (yalnızca politika aktarımı)
- Sadece referansa yönlendiren boş cevap (ör. "detaylar için ... bakınız")
"""

REWRITE_PROMPT = """\
Yanıtı yeniden yaz.

Kurallar:
1. Kullanıcı sorusuna doğrudan cevap ver.
2. "bakınız", "detay için şu sayfaya bakın" gibi referansa yönlendiren tek cümle formatı kullanma.
3. Sadece gerekli yerlerde citation ekle; aynı kaynağı gereksiz tekrar etme.
4. Bağlamda olmayan bilgi ekleme.

Kullanıcı sorusu:
{query}

Bağlam:
{context}

Mevcut yetersiz yanıt:
{answer}
"""

NO_CONTEXT_MSG = """\
Üzgünüm, bu soruyu güvenle yanıtlayacak kadar ilgili kaynak bulunamadı. \
Lütfen sorunuzu daha açık veya farklı şekilde ifade etmeyi deneyin ya da \
doğrudan **İK departmanıyla** iletişime geçin.
"""

_UNKNOWN_DOC = "Bilinmeyen Doküman"
_REFERENCE_ONLY_RE = re.compile(
    r"(bakınız|bakiniz|detay.*için|detay.*icin|see\s+.*page|refer\s+to)",
    re.IGNORECASE,
)


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
        self.max_generation_chunks = settings.max_generation_chunks
        self.max_sources_in_response = settings.max_sources_in_response

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

        selected_chunks = self._select_generation_chunks(retrieval_result.chunks)
        context_block = self._format_context(selected_chunks)
        sources = self._extract_sources(selected_chunks)
        messages = self._build_messages(query, context_block, history or [])

        response = llm_call(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        answer = self._coerce_text_response(response).strip()

        if self._needs_rewrite(answer):
            rewrite = llm_call(
                model=self.model,
                max_tokens=1500,
                system=SYSTEM_PROMPT,
                messages=[
                    {
                        "role": "user",
                        "content": REWRITE_PROMPT.format(
                            query=query,
                            context=context_block,
                            answer=answer,
                        ),
                    }
                ],
            )
            answer = self._coerce_text_response(rewrite).strip() or answer

        return GenerationResult(
            answer=answer,
            sources=sources,
            strategy_used=str(retrieval_result.strategy_used),
            grounded=True,
            queries_tried=retrieval_result.queries_tried,
        )

    # ------------------------------------------------------------------

    def _select_generation_chunks(self, chunks) -> list:
        """Keep generation context tight to reduce noisy or repetitive citations."""
        seen: set[tuple[str, int, str]] = set()
        selected: list = []
        for chunk in sorted(chunks, key=lambda c: c.score, reverse=True):
            key = (chunk.source, chunk.page_num, chunk.section)
            if key in seen:
                continue
            seen.add(key)
            selected.append(chunk)
            if len(selected) >= self.max_generation_chunks:
                break
        return selected or list(chunks[: self.max_generation_chunks])

    @staticmethod
    def _format_context(chunks) -> str:
        parts: list[str] = []
        for i, chunk in enumerate(chunks, start=1):
            doc_name = Path(chunk.source).stem if chunk.source else _UNKNOWN_DOC
            section_label = chunk.breadcrumb or chunk.section
            # Use full parent section for richer LLM context;
            # child chunk text is preserved in sources for UI highlighting
            context_text = chunk.parent_text if chunk.parent_text else chunk.text
            parts.append(
                f"[Kaynak {i}: {doc_name} | Bölüm: {section_label} | Sayfa: {chunk.page_num}]\n"
                f"{context_text}"
            )
        return "\n\n---\n\n".join(parts)

    def _extract_sources(self, chunks) -> list[dict]:
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
        return sorted(
            sources, key=lambda s: s["score"], reverse=True
        )[: self.max_sources_in_response]

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

    @staticmethod
    def _coerce_text_response(response) -> str:
        if isinstance(response, str):
            return response
        content = getattr(response, "content", None)
        if isinstance(content, list) and content:
            text = getattr(content[0], "text", None)
            if isinstance(text, str):
                return text
        return str(response or "")

    @staticmethod
    def _needs_rewrite(answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        has_reference_only_phrase = bool(_REFERENCE_ONLY_RE.search(text))
        has_citation = ("[") in text and ("s." in text or "p." in text)
        # Very short directional answers are usually low quality for this project.
        is_short = len(text) < 220
        return has_reference_only_phrase and (is_short or not has_citation)
