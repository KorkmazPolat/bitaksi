from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Provider: "openai" or "gemini"
    llm_provider: str = "openai"

    llm_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"
    embedding_model: str = "paraphrase-multilingual-mpnet-base-v2"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_raw: str = "hr_docs_raw"
    chroma_collection_relatives: str = "hr_docs_relatives"

    top_k: int = 8
    similarity_threshold: float = 0.35
    relatives_score_threshold: float = 0.35   # lower bar for Q→chunk matching
    fallback_top_k: int = 12

    # Hybrid retrieval
    use_bm25: bool = True
    use_reranking: bool = True
    rerank_model: str = "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1"
    rerank_candidate_k: int = 30   # candidates fed to cross-encoder
    rerank_final_k: int = 8        # output after reranking
    rrf_k: int = 60                # RRF constant (standard value)

    # Chunking — hierarchical semantic chunker
    chunk_max_tokens: int = 512          # child chunk ceiling (tokens)
    chunk_overlap_tokens: int = 128      # sliding-window overlap when hard-splitting
    chunk_min_tokens: int = 50           # merge groups smaller than this
    semantic_split_percentile: int = 95  # percentile of cosine-distance for breakpoints
    use_contextual_enrichment: bool = True
    contextual_enrichment_min_tokens: int = 120
    contextual_enrichment_max_chars: int = 1200

    # Legacy word-based params kept for backward compat
    chunk_size: int = 800
    chunk_overlap: int = 100

    eval_recall_k: int = 5

    # Ingestion
    batch_upsert_size: int = 100
    docx_page_paragraph_count: int = 50
    min_chunk_length_for_relatives: int = 20
    chunk_preview_length: int = 1500        # chars sent to relative-question LLM
    min_relative_question_length: int = 12
    max_relative_question_length: int = 220

    # Generation
    max_history_turns: int = 6
    max_query_length: int = 2000
    max_upload_size_mb: int = 100
    max_generation_chunks: int = 5
    max_sources_in_response: int = 5

    # Evaluation
    faithfulness_context_limit: int = 4000

    # Retrieval ranking
    lexical_overlap_weight: float = 0.15

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
