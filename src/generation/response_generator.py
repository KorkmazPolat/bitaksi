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
from src.utils import llm

logger = logging.getLogger(__name__)

SYSTEM_PROMPT = """\
Sen dokumanlarla sinirli bir IK sohbet asistanisin.

Gorevin:
- Kullaniciyla dogal bir diyalog kur.
- Butun politik, surecsel, sayisal veya hakka dair ifadelerini sadece verilen baglamdan uret.
- Konusma gecmisini sadece referansi cozmeye yardimci olmak icin kullan; gecmis mesajlari bagimsiz kaynak sayma.

Kesin kurallar:
1. Baglam disindan bilgi uretme.
2. Genel IK bilgisi, tahmin, varsayim veya sirket disi norm ekleme.
3. Sadece su yanit turlerinden birini sec:
   - `answer`: Baglam soruyu cevaplamak icin yeterliyse kisa ve net cevap ver. Onemli iddialari `[dokuman, s.X]` formatinda cite et.
   - `abstain`: Baglamda cevap yoksa bunu acikca soyle. Uydurma bilgi verme. Gerekirse IK ile iletisime gecilmesini oner.
   - `clarify`: Soru muglaksa ve mevcut baglamla hangi politika veya durumun kastedildigi guvenle anlasilamiyorsa tek bir kisa netlestirici soru sor.
4. `clarify` ve `abstain` durumunda citation ekleme.
5. `answer` durumunda cevap en fazla 2 kisa paragraf veya 3 madde olsun.
6. Cevabi kullanicinin dilinde ver.

Sadece gecerli JSON dondur:
{
  "answer_type": "answer" | "abstain" | "clarify",
  "answer": "<kullaniciya gidecek metin>"
}
"""

REWRITE_PROMPT = """\
Yanıtı yeniden yaz.

Kurallar:
1. Kullanıcı sorusuna doğrudan cevap ver.
2. "bakınız", "detay için şu sayfaya bakın" gibi referansa yönlendiren tek cümle formatı kullanma.
3. Sadece gerekli yerlerde citation ekle; aynı kaynağı gereksiz tekrar etme.
4. Bağlamda olmayan bilgi ekleme.
5. Duz metin don; JSON donme.

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
_ABSTAIN_RE = re.compile(
    r"("
    r"bilgi (bulunamadi|bulunamıyor|bulunamiyor|bulunmamaktadir|bulunmuyor|yer almiyor|yer almamaktadir)"
    r"|bağlamda mevcut değil|baglamda mevcut degil"
    r"|dokümanlarda .* yok|dokumanlarda .* yok"
    r"|ik departman[ıi]yla iletişim|ik departman[ıi]yla iletisim"
    r"|ik ile iletişim|ik ile iletisim"
    r")",
    re.IGNORECASE,
)
_CLARIFY_RE = re.compile(
    r"(neyi kastettiniz|hangi (konu|politika|izin|dok[üu]man|belge|durum|s[üu]re[cç]|harcama|seyahat|talep)|biraz daha netle[sş]tir|biraz daha a[cç]ar m[ıi]s[ıi]n[ıi]z|sorunuzu biraz daha)",
    re.IGNORECASE,
)
_DEICTIC_QUERY_RE = re.compile(
    r"\b(bu|buna|bunu|bunlar|o|onu|onun|orada|burada|boyle|böyle|bu durumda|o durumda)\b",
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
    citations: list[dict]
    strategy_used: str
    grounded: bool
    answer_type: Literal["answer", "abstain", "clarify"] = "answer"
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
                citations=[],
                strategy_used=str(retrieval_result.strategy_used),
                grounded=False,
                answer_type="abstain",
                queries_tried=retrieval_result.queries_tried,
            )

        selected_chunks = self._select_generation_chunks(retrieval_result.chunks)
        context_block = self._format_context(selected_chunks)
        messages = self._build_messages(query, context_block, history or [])

        response = llm.llm_call(
            model=self.model,
            max_tokens=1500,
            system=SYSTEM_PROMPT,
            messages=messages,
        )
        raw_text = self._coerce_text_response(response).strip()
        parsed_type, answer = self._parse_model_output(raw_text)
        answer = answer.strip()

        if parsed_type == "answer" and self._needs_rewrite(answer):
            rewrite = llm.llm_call(
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

        sources = self._extract_sources(selected_chunks, answer) if parsed_type == "answer" else []
        citations = self._extract_citations(answer, sources) if parsed_type == "answer" else []
        answer_type = self._finalize_answer_type(
            query=query,
            parsed_type=parsed_type,
            answer=answer,
            citations=citations,
        )
        grounded = answer_type == "answer"
        if not grounded:
            sources = []
            citations = []

        return GenerationResult(
            answer=answer,
            sources=sources,
            citations=citations,
            strategy_used=str(retrieval_result.strategy_used),
            grounded=grounded,
            answer_type=answer_type,
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

    def _extract_sources(self, chunks, answer: str) -> list[dict]:
        seen: set[str] = set()
        sources: list[dict] = []
        for chunk in chunks:
            doc_name = Path(chunk.source).stem if chunk.source else _UNKNOWN_DOC
            key = chunk.chunk_id or f"{doc_name}:{chunk.page_num}:{chunk.section}"
            if key not in seen:
                seen.add(key)
                highlight_text = self._extract_highlight_text(answer, chunk.text)
                sources.append(
                    {
                        "document": doc_name,
                        "page": chunk.page_num,
                        "section": chunk.section,
                        "score": round(chunk.score, 3),
                        "chunk_text": chunk.text,   # used by frontend for highlighting
                        "highlight_text": highlight_text,
                    }
                )
        return sorted(
            sources, key=lambda s: s["score"], reverse=True
        )[: self.max_sources_in_response]

    @classmethod
    def _extract_highlight_text(cls, answer: str, chunk_text: str) -> str:
        if not chunk_text:
            return ""

        chunk_sentences = cls._split_sentences(chunk_text)
        if not chunk_sentences:
            return chunk_text.strip()

        answer_sentences = cls._split_sentences(answer)
        if not answer_sentences:
            answer_sentences = [answer]

        candidate_windows = cls._build_sentence_windows(chunk_sentences)

        best_text = ""
        best_score = 0.0
        for candidate_text in candidate_windows:
            for answer_sentence in answer_sentences:
                score = cls._sentence_overlap_score(answer_sentence, candidate_text)
                if score > best_score:
                    best_score = score
                    best_text = candidate_text

        if best_text and best_score >= 0.2:
            return cls._clean_highlight_text(best_text)

        return cls._clean_highlight_text(" ".join(chunk_sentences[:2]))

    @staticmethod
    def _build_sentence_windows(sentences: list[str]) -> list[str]:
        windows: list[str] = []
        total = len(sentences)
        for start in range(total):
            for width in (1, 2):
                end = start + width
                if end <= total:
                    windows.append(" ".join(sentences[start:end]).strip())
        return windows

    @staticmethod
    def _clean_highlight_text(text: str) -> str:
        cleaned = re.sub(r"\[[^\]]+\]", "", text or "")
        cleaned = re.sub(r"\s+", " ", cleaned).strip()
        return cleaned

    @staticmethod
    def _split_sentences(text: str) -> list[str]:
        cleaned = re.sub(r"\s+", " ", (text or "").strip())
        if not cleaned:
            return []
        parts = re.split(r"(?<=[.!?])\s+", cleaned)
        return [part.strip() for part in parts if part.strip()]

    @staticmethod
    def _tokenize_for_overlap(text: str) -> set[str]:
        return {
            token
            for token in re.findall(r"\w+", (text or "").lower())
            if len(token) >= 3 and not token.isdigit()
        }

    @classmethod
    def _sentence_overlap_score(cls, answer_sentence: str, chunk_sentence: str) -> float:
        answer_tokens = cls._tokenize_for_overlap(answer_sentence)
        chunk_tokens = cls._tokenize_for_overlap(chunk_sentence)
        if not answer_tokens or not chunk_tokens:
            return 0.0
        overlap = answer_tokens & chunk_tokens
        if not overlap:
            return 0.0
        return len(overlap) / max(1, len(answer_tokens))

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
                f"## Konusma Notu\n"
                f"Konusma gecmisi sadece referans cozumleme icindir; "
                f"olgusal kaynak olarak kullanma.\n\n"
                f"## Ilgili IK Dokumani Baglami\n\n{context}\n\n"
                f"---\n\n## Calisan Sorusu\n\n{query}"
            ),
        })
        return messages

    @classmethod
    def _extract_citations(cls, answer: str, sources: list[dict]) -> list[dict]:
        pattern = re.compile(r"\[([^,\]]+),\s*s\.(\d+)\]")
        citations: list[dict] = []

        for match in pattern.finditer(answer or ""):
            doc = (match.group(1) or "").strip()
            page = int(match.group(2) or "1")
            context = cls._extract_citation_context(answer, match.start(), match.end())
            source_index = cls._match_citation_to_source(doc, page, context, sources)
            evidence_text = ""
            if 0 <= source_index < len(sources):
                source = sources[source_index]
                evidence_text = cls._extract_highlight_text(
                    context,
                    source.get("chunk_text", ""),
                )
            citations.append(
                {
                    "document": doc,
                    "page": page,
                    "context": context,
                    "source_index": source_index,
                    "evidence_text": evidence_text,
                }
            )
        return citations

    @staticmethod
    def _extract_citation_context(answer: str, start: int, end: int) -> str:
        if not answer:
            return ""
        sentence_start = max(
            answer.rfind(".", 0, start),
            answer.rfind("!", 0, start),
            answer.rfind("?", 0, start),
            answer.rfind("\n", 0, start),
        )
        candidates = [
            idx for idx in (
                answer.find(".", end),
                answer.find("!", end),
                answer.find("?", end),
                answer.find("\n", end),
            ) if idx != -1
        ]
        sentence_end = min(candidates) if candidates else len(answer)
        return answer[sentence_start + 1:sentence_end].strip()

    @classmethod
    def _match_citation_to_source(
        cls,
        document: str,
        page: int,
        context: str,
        sources: list[dict],
    ) -> int:
        doc_key = (document or "").strip().lower()
        candidates = [
            (idx, source)
            for idx, source in enumerate(sources)
            if (source.get("document", "").strip().lower() == doc_key)
            and int(source.get("page", 0) or 0) == int(page or 0)
        ]
        if not candidates:
            return -1
        if len(candidates) == 1:
            return candidates[0][0]

        best_idx = candidates[0][0]
        best_score = -1.0
        for idx, source in candidates:
            snippet = source.get("highlight_text") or source.get("chunk_text") or ""
            score = cls._sentence_overlap_score(context, snippet)
            if score > best_score:
                best_idx = idx
                best_score = score
        return best_idx

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

    @staticmethod
    def _is_abstain_answer(answer: str, citations: list[dict]) -> bool:
        text = (answer or "").strip()
        if not text:
            return True
        if text == NO_CONTEXT_MSG.strip():
            return True
        if citations:
            return False
        return bool(_ABSTAIN_RE.search(text))

    @classmethod
    def _is_clarify_answer(cls, query: str, answer: str) -> bool:
        text = (answer or "").strip()
        if not text:
            return False
        if _CLARIFY_RE.search(text):
            return True
        return (
            text.endswith("?")
            and len(text) <= 220
            and cls._query_needs_clarification(query)
        )

    @staticmethod
    def _query_needs_clarification(query: str) -> bool:
        text = (query or "").strip()
        if not text:
            return True
        tokens = re.findall(r"\w+", text.lower())
        if len(tokens) <= 3:
            return True
        return bool(_DEICTIC_QUERY_RE.search(text))

    @staticmethod
    def _normalize_answer_type(value: str | None) -> Literal["answer", "abstain", "clarify"] | None:
        normalized = (value or "").strip().lower()
        if normalized in {"answer", "abstain", "clarify"}:
            return normalized
        return None

    def _parse_model_output(
        self,
        raw_text: str,
    ) -> tuple[Literal["answer", "abstain", "clarify"], str]:
        parsed_type: Literal["answer", "abstain", "clarify"] | None = None
        parsed_answer = raw_text
        try:
            parsed = llm.parse_llm_json(raw_text)
        except Exception:
            parsed = None

        if isinstance(parsed, dict):
            parsed_type = self._normalize_answer_type(parsed.get("answer_type"))
            candidate_answer = parsed.get("answer")
            if isinstance(candidate_answer, str) and candidate_answer.strip():
                parsed_answer = candidate_answer.strip()

        return parsed_type or "answer", parsed_answer

    def _finalize_answer_type(
        self,
        *,
        query: str,
        parsed_type: Literal["answer", "abstain", "clarify"],
        answer: str,
        citations: list[dict],
    ) -> Literal["answer", "abstain", "clarify"]:
        if parsed_type == "clarify" or self._is_clarify_answer(query, answer):
            return "clarify"
        if parsed_type == "abstain" or self._is_abstain_answer(answer, citations):
            return "abstain"
        return "answer"
