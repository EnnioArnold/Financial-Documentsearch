"""Request validation and response shapes of the HTTP API (pydantic v2)."""

from __future__ import annotations

import html
import re
from datetime import date
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from finsearch.contracts import Citation, RagAnswer, SearchFilters, SearchHit, SearchResult

MAX_QUERY_CHARS = 500
MAX_K = 50
DEFAULT_K = 10
MAX_PAGE_SIZE = 100
SNIPPET_CHARS = 300
SLUG = r"^[a-z0-9_-]+$"
MAX_SLUG_CHARS = 40

Query = Annotated[
    str, StringConstraints(strip_whitespace=True, min_length=1, max_length=MAX_QUERY_CHARS)
]
Slug = Annotated[str, StringConstraints(pattern=SLUG, max_length=MAX_SLUG_CHARS)]
MethodName = Literal["bm25", "knn", "hybrid", "hybrid_rerank"]
K = Annotated[int, Field(ge=1, le=MAX_K)]

_EM_TAG = re.compile(r"(</?em>)")


def safe_highlight(raw: str) -> str:
    """Defence in depth: keep only <em>/</em> as markup, everything else stays escaped text.

    The search layer already HTML-escapes; unescape-then-escape is idempotent for such input and
    neutralises any raw tag that slipped through.
    """
    parts = _EM_TAG.split(raw)
    return "".join(p if _EM_TAG.fullmatch(p) else html.escape(html.unescape(p)) for p in parts)


class Filters(BaseModel):
    """Optional filters shared by /search, /compare and /ask."""

    model_config = ConfigDict(populate_by_name=True, extra="forbid")

    source: Slug | None = None
    date_from: date | None = Field(None, alias="from")
    date_to: date | None = Field(None, alias="to")
    lang: Literal["de", "en"] | None = None
    doc_type: Slug | None = None

    @model_validator(mode="before")
    @classmethod
    def _blank_is_none(cls, data: Any) -> Any:
        """HTML forms send empty strings for unset filters."""
        if isinstance(data, dict):
            return {k: (None if v == "" else v) for k, v in data.items()}
        return data

    @model_validator(mode="after")
    def _check_range(self) -> Filters:
        if self.date_from and self.date_to and self.date_from > self.date_to:
            raise ValueError("'from' darf nicht nach 'to' liegen")
        return self

    def to_search_filters(self) -> SearchFilters:
        return SearchFilters(
            sources=[self.source] if self.source else [],
            lang=self.lang,
            doc_type=self.doc_type,
            date_from=self.date_from,
            date_to=self.date_to,
        )


class SearchParams(Filters):
    q: Query
    method: MethodName = "hybrid"
    k: K = DEFAULT_K

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class CompareParams(Filters):
    q: Query
    k: K = DEFAULT_K

    model_config = ConfigDict(populate_by_name=True, extra="ignore")


class DocumentsParams(BaseModel):
    source: Slug | None = None
    limit: int = Field(50, ge=1, le=MAX_PAGE_SIZE)
    offset: int = Field(0, ge=0)


class AskRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    question: Query
    method: MethodName = "hybrid"
    k: K = 6
    filters: Filters = Field(default_factory=lambda: Filters.model_validate({}))


# ---- responses ---------------------------------------------------------------------------------


class HitOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    date: date | None
    page: int
    url: str
    lang: str
    doc_type: str
    score: float
    rank: int
    method: str
    highlights: list[str]  # HTML-escaped text; the only markup is <em>
    snippet: str  # plain-text excerpt, not HTML
    text: str  # full chunk text (plain)

    @classmethod
    def from_hit(cls, hit: SearchHit) -> HitOut:
        c = hit.chunk
        flat = " ".join(c.text.split())
        snippet = (
            flat if len(flat) <= SNIPPET_CHARS else flat[:SNIPPET_CHARS].rsplit(" ", 1)[0] + "…"
        )
        return cls(
            chunk_id=c.chunk_id,
            doc_id=c.doc_id,
            title=c.title,
            source=c.source,
            date=c.date,
            page=c.page,
            url=c.url,
            lang=c.lang,
            doc_type=c.doc_type,
            score=hit.score,
            rank=hit.rank,
            method=hit.method,
            highlights=[safe_highlight(h) for h in hit.highlights],
            snippet=snippet,
            text=c.text,
        )


class SearchResponse(BaseModel):
    query: str
    method: str
    k: int
    n_hits: int
    total_ms: float
    timings_ms: dict[str, float]
    hits: list[HitOut]

    @classmethod
    def from_result(cls, result: SearchResult, k: int) -> SearchResponse:
        return cls(
            query=result.query,
            method=result.method,
            k=k,
            n_hits=len(result.hits),
            total_ms=result.total_ms,
            timings_ms=result.timings_ms,
            hits=[HitOut.from_hit(h) for h in result.hits],
        )


class CompareResponse(BaseModel):
    query: str
    k: int
    results: dict[str, SearchResponse]  # keyed by method: bm25, knn, hybrid


class CitationOut(BaseModel):
    n: int
    chunk_id: str
    title: str
    page: int
    url: str

    @classmethod
    def from_citation(cls, c: Citation) -> CitationOut:
        return cls(n=c.n, chunk_id=c.chunk_id, title=c.title, page=c.page, url=c.url)


class ChunkOut(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    source: str
    date: date | None
    page: int
    url: str
    lang: str
    text: str


class AskResponse(BaseModel):
    question: str
    answer: str
    found: bool
    citations: list[CitationOut]
    used_chunks: list[ChunkOut]  # used_chunks[n - 1] is source [n]
    confidence_note: str
    provider: str
    dropped_citations: list[int]
    timings_ms: dict[str, float]

    @classmethod
    def from_answer(cls, question: str, a: RagAnswer) -> AskResponse:
        return cls(
            question=question,
            answer=a.answer,
            found=a.found,
            citations=[CitationOut.from_citation(c) for c in a.citations],
            used_chunks=[
                ChunkOut(**{f: getattr(c, f) for f in ChunkOut.model_fields}) for c in a.used_chunks
            ],
            confidence_note=a.confidence_note,
            provider=a.provider,
            dropped_citations=a.dropped_citations,
            timings_ms=a.timings_ms,
        )


class DocumentOut(BaseModel):
    doc_id: str
    title: str | None = None
    source: str | None = None
    date: str | None = None
    url: str | None = None
    lang: str | None = None
    doc_type: str | None = None
    n_chunks: int


class DocumentsResponse(BaseModel):
    limit: int
    offset: int
    items: list[DocumentOut]
