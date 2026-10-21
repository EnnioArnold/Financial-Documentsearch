"""Shared data types and interfaces. Every module depends on this file, never on each other's internals."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Literal, Protocol, runtime_checkable

Lang = Literal["de", "en", "other"]
Method = Literal["bm25", "knn", "hybrid", "hybrid_rerank"]
METHODS: tuple[str, ...] = ("bm25", "knn", "hybrid", "hybrid_rerank")


@dataclass(frozen=True)
class DocumentMeta:
    """One source document (before chunking)."""

    doc_id: str  # stable: first 16 hex chars of sha256(url)
    source: str  # source family key from sources.yaml, e.g. "snb"
    title: str
    url: str
    lang: Lang
    doc_type: str  # speech | annual_report | market_comment | other
    date: date | None = None
    n_pages: int = 1
    content_hash: str = ""  # sha256 of raw bytes, for dedup / change detection


@dataclass(frozen=True)
class Chunk:
    """Smallest indexed unit. chunk_id = f"{doc_id}-{ord:04d}" (deterministic => idempotent ingest)."""

    chunk_id: str
    doc_id: str
    ord: int
    text: str
    page: int  # 1-based page where the chunk starts
    source: str
    title: str
    url: str
    lang: Lang
    doc_type: str
    date: date | None = None


@dataclass(frozen=True)
class SearchFilters:
    sources: list[str] = field(default_factory=list)
    lang: Lang | None = None
    doc_type: str | None = None
    date_from: date | None = None
    date_to: date | None = None


@dataclass(frozen=True)
class SearchHit:
    chunk: Chunk
    score: float
    rank: int  # 1-based
    method: str
    highlights: list[str] = field(default_factory=list)  # snippets with <em>..</em>


@dataclass
class SearchResult:
    query: str
    method: str
    hits: list[SearchHit]
    total_ms: float
    timings_ms: dict[str, float] = field(
        default_factory=dict
    )  # bm25 / knn / embed / fusion / rerank


@runtime_checkable
class Embedder(Protocol):
    """Text -> L2-normalised vectors. Implementations add the e5 'query: '/'passage: ' prefixes."""

    dim: int

    def embed_queries(self, texts: list[str]) -> list[list[float]]: ...
    def embed_passages(self, texts: list[str]) -> list[list[float]]: ...


@runtime_checkable
class Reranker(Protocol):
    def rerank(self, query: str, hits: list[SearchHit], top_k: int) -> list[SearchHit]: ...


@runtime_checkable
class Searcher(Protocol):
    def search(
        self, query: str, method: str = "hybrid", k: int = 10, filters: SearchFilters | None = None
    ) -> SearchResult: ...


# ---- RAG ----------------------------------------------------------------------------------


@dataclass(frozen=True)
class ChatMessage:
    role: Literal["system", "user", "assistant"]
    content: str


@dataclass(frozen=True)
class LLMResponse:
    text: str
    model: str
    latency_ms: float


@runtime_checkable
class LLMClient(Protocol):
    name: str

    def generate(self, messages: list[ChatMessage], **opts: object) -> LLMResponse:
        """Raise LLMError on timeout / HTTP / auth problems (never return garbage)."""
        ...


class LLMError(RuntimeError):
    pass


NOT_FOUND_TEXT = "In den Quellen nicht gefunden"


@dataclass(frozen=True)
class Citation:
    n: int  # the [n] marker in the answer
    chunk_id: str
    title: str
    page: int
    url: str


@dataclass
class RagAnswer:
    answer: str
    citations: list[Citation]
    used_chunks: list[Chunk]
    confidence_note: str
    found: bool  # False => answer is NOT_FOUND_TEXT
    provider: str = "extractive"
    dropped_citations: list[int] = field(default_factory=list)  # [n] that pointed nowhere
    timings_ms: dict[str, float] = field(default_factory=dict)
