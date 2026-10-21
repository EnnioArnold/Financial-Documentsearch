"""All settings in one place. Values come from environment / .env (see .env.example)."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT = Path(__file__).resolve().parents[2]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    es_url: str = "http://127.0.0.1:9200"
    es_index_alias: str = "finsearch"
    es_index_version: str = "v1"
    es_queries_index: str = "finsearch-queries"
    kibana_url: str = "http://127.0.0.1:5601"

    embedding_model: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    embedding_batch_size: int = 32
    reranker_model: str = "BAAI/bge-reranker-base"

    # search tuning (documented + evaluated in docs/DECISIONS.md)
    rrf_k: int = 60
    candidate_pool: int = 50  # per-method candidates fed into fusion
    rerank_pool: int = 30
    boost_de: float = 1.0
    boost_en: float = 1.0
    boost_text: float = 0.5

    # chunking
    chunk_target_tokens: int = 400
    chunk_overlap_ratio: float = 0.12

    # ingest
    sources_file: Path = ROOT / "sources.yaml"
    raw_dir: Path = ROOT / "data" / "raw"
    processed_dir: Path = ROOT / "data" / "processed"
    ingest_user_agent: str = "FinSearch-Portfolio/0.1 (research)"
    ingest_delay_s: float = 1.0

    # rag
    llm_provider: str = "extractive"
    llm_model: str = ""
    llm_timeout_s: float = 30.0
    ollama_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "llama3.1"
    gemini_api_key: str = ""
    groq_api_key: str = ""
    anthropic_api_key: str = ""
    rag_top_k: int = 6
    rag_min_score: float = 0.0  # evidence gate, tuned in Phase 3

    # observability
    log_raw_queries: bool = False

    @property
    def index_name(self) -> str:
        return f"{self.es_index_alias}-{self.es_index_version}"


@lru_cache
def get_settings() -> Settings:
    return Settings()
