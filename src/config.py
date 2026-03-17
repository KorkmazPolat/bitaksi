from pydantic_settings import BaseSettings
from functools import lru_cache


class Settings(BaseSettings):
    anthropic_api_key: str = ""

    llm_model: str = "claude-sonnet-4-6"
    vision_model: str = "claude-sonnet-4-6"
    embedding_model: str = "all-MiniLM-L6-v2"

    chroma_persist_dir: str = "./data/chroma"
    chroma_collection_raw: str = "hr_docs_raw"
    chroma_collection_relatives: str = "hr_docs_relatives"

    top_k: int = 5
    similarity_threshold: float = 0.6
    fallback_top_k: int = 8

    chunk_size: int = 800
    chunk_overlap: int = 100

    eval_recall_k: int = 5

    class Config:
        env_file = ".env"


@lru_cache()
def get_settings() -> Settings:
    return Settings()
