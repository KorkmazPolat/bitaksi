from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    openai_api_key: str = ""
    gemini_api_key: str = ""

    # Provider: "openai" or "gemini"
    llm_provider: str = "openai"

    llm_model: str = "gpt-4o"
    vision_model: str = "gpt-4o"
    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_raw: str = "hr_docs_raw"
    chroma_collection_relatives: str = "hr_docs_relatives"

    top_k: int = 5
    similarity_threshold: float = 0.6
    relatives_score_threshold: float = 0.5   # lower bar for Q→chunk matching
    fallback_top_k: int = 8

    # Chunking — hierarchical semantic chunker
    chunk_max_tokens: int = 512          # child chunk ceiling (tokens)
    chunk_overlap_tokens: int = 128      # sliding-window overlap when hard-splitting
    chunk_min_tokens: int = 50           # merge groups smaller than this
    semantic_split_percentile: int = 95  # percentile of cosine-distance for breakpoints
    use_contextual_enrichment: bool = False  # LLM-generated per-chunk context (slow, opt-in)

    # Legacy word-based params kept for backward compat
    chunk_size: int = 800
    chunk_overlap: int = 100

    eval_recall_k: int = 5

    # Ingestion
    batch_upsert_size: int = 100
    docx_page_paragraph_count: int = 50
    min_chunk_length_for_relatives: int = 50
    chunk_preview_length: int = 1500        # chars sent to relative-question LLM

    # Generation
    max_history_turns: int = 6
    max_query_length: int = 2000
    max_upload_size_mb: int = 100

    # Evaluation
    faithfulness_context_limit: int = 4000

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
