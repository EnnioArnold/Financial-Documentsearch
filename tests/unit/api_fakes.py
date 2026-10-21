"""Offline fakes for the API tests (no Elasticsearch, no models)."""

from __future__ import annotations

from datetime import date
from typing import Any

from elastic_transport import ApiResponseMeta, HttpHeaders, NodeConfig
from elasticsearch import ConnectionError as EsConnectionError
from elasticsearch import NotFoundError
from fastapi.testclient import TestClient
from rag_fakes import FakeLLM, settings

from finsearch.api.deps import Services
from finsearch.api.main import create_app
from finsearch.config import Settings
from finsearch.contracts import Chunk, LLMClient, SearchFilters, SearchHit, SearchResult

XSS = "<script>alert(1)</script>"


def chunk(n: int, text: str, **kw: Any) -> Chunk:
    base: dict[str, Any] = {
        "chunk_id": f"doc{n}-0000",
        "doc_id": f"doc{n}",
        "ord": 0,
        "text": text,
        "page": n,
        "source": "snb",
        "title": f"Titel {n}",
        "url": f"https://example.org/doc{n}.pdf",
        "lang": "de",
        "doc_type": "speech",
        "date": date(2024, 6, 20),
    }
    base.update(kw)
    return Chunk(**base)


CHUNKS = [
    chunk(1, "Die Nationalbank senkt den Leitzins im Juni 2024 auf 1,25 Prozent."),
    chunk(2, "Die Inflation liegt im Zielbereich der Nationalbank.", title="Bericht"),
    chunk(3, f"Gefaehrlicher Text {XSS} im Dokument.", title=f"Boese {XSS}"),
]


def not_found_error() -> NotFoundError:
    meta = ApiResponseMeta(404, "1.1", HttpHeaders(), 0.0, NodeConfig("http", "x", 9200))
    return NotFoundError("index_not_found_exception", meta, {"error": "no such index"})


class FakeEs:
    def __init__(self, count: int = 3, error: Exception | None = None) -> None:
        self._count, self.error = count, error

    def options(self, **_: Any) -> FakeEs:
        return self

    def count(self, index: str) -> dict[str, int]:
        if self.error:
            raise self.error
        return {"count": self._count}


class FakeSearchService:
    """Stands in for SearchService: records calls, can fail on demand."""

    def __init__(self, es: FakeEs | None = None) -> None:
        self.es = es or FakeEs()
        self.index = "fake-index"
        self.calls: list[tuple[str, str, int, SearchFilters | None]] = []
        self.error: Exception | None = None  # raised by search()/stats()/list_documents()
        self.rerank_error: Exception | None = None  # raised only for hybrid_rerank

    def search(
        self, query: str, method: str = "hybrid", k: int = 10, filters: SearchFilters | None = None
    ) -> SearchResult:
        self.calls.append((query, method, k, filters))
        if self.error:
            raise self.error
        if method == "hybrid_rerank" and self.rerank_error:
            raise self.rerank_error
        hits = [
            SearchHit(c, 1.0 / i, i, method, [f"Die <em>Nationalbank</em> {XSS} &amp; mehr"])
            for i, c in enumerate(CHUNKS[:k], 1)
        ]
        timings = {"bm25": 1.5, "knn": 2.5, "embed": 3.5, "fusion": 0.5}
        return SearchResult(query, method, hits, 9.0, timings)

    def stats(self) -> dict[str, Any]:
        if self.error:
            raise self.error
        return {"documents": 3, "chunks": 3, "by_source": {"snb": {"chunks": 3, "documents": 3}}}

    def list_documents(
        self, source: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        if self.error:
            raise self.error
        self.calls.append(("documents", str(source), limit, None))
        row = {"doc_id": "doc1", "title": "Titel 1", "source": "snb", "date": "2024-06-20"}
        return [
            {
                **row,
                "url": "https://example.org/1",
                "lang": "de",
                "doc_type": "speech",
                "n_chunks": 2,
            }
        ][offset:]


class FakeQueryLogger:
    def __init__(self) -> None:
        self.records: list[dict[str, Any]] = []
        self.ensured = self.closed = False

    def ensure_index(self) -> bool:
        self.ensured = True
        return True

    def record(self, **kw: Any) -> None:
        self.records.append(kw)

    def close(self) -> None:
        self.closed = True


class Env:
    """Everything a test may want to inspect."""

    def __init__(self, cfg: Settings, llm: LLMClient | None = None) -> None:
        self.search = FakeSearchService()
        self.logger = FakeQueryLogger()
        kwargs: dict[str, Any] = {"search_factory": lambda s: self.search}
        if llm is not None:
            kwargs["llm_factory"] = lambda s: llm
        self.app = create_app(cfg, Services(cfg, **kwargs), self.logger)  # type: ignore[arg-type]
        self.client = TestClient(self.app, raise_server_exceptions=False)


def make_env(llm: LLMClient | None = None, **cfg: Any) -> Env:
    cfg.setdefault("api_rate_limit", 0)  # off unless a test is about the limit
    return Env(settings(**cfg), llm)


__all__ = ["CHUNKS", "XSS", "EsConnectionError", "FakeLLM", "make_env", "not_found_error"]
