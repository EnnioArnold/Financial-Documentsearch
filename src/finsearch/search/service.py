"""SearchService: the one entry point the API / RAG / eval code uses (implements `Searcher`)."""

from __future__ import annotations

import html
import logging
import time
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from finsearch.config import Settings, get_settings
from finsearch.contracts import (
    METHODS,
    Chunk,
    Embedder,
    Reranker,
    SearchFilters,
    SearchHit,
    SearchResult,
)
from finsearch.search.bm25 import bm25_search
from finsearch.search.embedder import get_embedder
from finsearch.search.es_client import get_es
from finsearch.search.filters import SOURCE_EXCLUDES, normalize_query
from finsearch.search.fusion import rrf_fuse
from finsearch.search.mapping import doc_to_chunk
from finsearch.search.rerank import get_reranker
from finsearch.search.vector import knn_search

log = logging.getLogger(__name__)

MAX_K = 100
SNIPPET_CHARS = 200
MAX_AGG_BUCKETS = 1000
DOC_SOURCE_FIELDS = ["doc_id", "title", "source", "date", "url", "lang", "doc_type"]


def _ms(start: float) -> float:
    return round((time.perf_counter() - start) * 1000, 3)


def fallback_snippet(text: str, max_chars: int = SNIPPET_CHARS) -> str:
    """Plain leading excerpt, HTML-escaped like real highlights so `highlights` is always safe markup."""
    flat = " ".join(text.split())
    if len(flat) > max_chars:
        flat = flat[:max_chars].rsplit(" ", 1)[0] + "…"
    return html.escape(flat)


class SearchService:
    def __init__(
        self,
        es: Elasticsearch,
        embedder: Embedder,
        reranker: Reranker | None = None,
        settings: Settings | None = None,
        index: str | None = None,
    ) -> None:
        self.es = es
        self.embedder = embedder
        self.reranker = reranker
        self.settings = settings or get_settings()
        self.index = (
            index or self.settings.es_index_alias
        )  # read through the alias => re-index safe

    # ---- search -----------------------------------------------------------------------------

    def search(
        self, query: str, method: str = "hybrid", k: int = 10, filters: SearchFilters | None = None
    ) -> SearchResult:
        """Run one retrieval method. Blank queries return an empty result; bad arguments raise ValueError."""
        if method not in METHODS:
            raise ValueError(f"unknown method {method!r}; expected one of {METHODS}")
        if method == "hybrid_rerank" and self.reranker is None:
            raise ValueError("hybrid_rerank needs a reranker (pass reranker=... to SearchService)")
        if not 1 <= k <= MAX_K:
            raise ValueError(f"k must be between 1 and {MAX_K}, got {k}")
        start = time.perf_counter()
        q = normalize_query(query)
        timings: dict[str, float] = {}
        hits: list[SearchHit] = []
        if q:
            if method == "bm25":
                hits = self._bm25(q, filters, k, timings)
            elif method == "knn":
                hits = self._knn(q, filters, k, timings)
            else:
                hits = self._hybrid(q, filters, k, timings, rerank=method == "hybrid_rerank")
        return SearchResult(
            query=query, method=method, hits=hits, total_ms=_ms(start), timings_ms=timings
        )

    def _bm25(
        self, q: str, filters: SearchFilters | None, k: int, timings: dict[str, float]
    ) -> list[SearchHit]:
        t = time.perf_counter()
        raw = bm25_search(self.es, self.index, q, filters, k, self.settings)
        timings["bm25"] = _ms(t)
        return [
            SearchHit(c, score, i, "bm25", hl or [fallback_snippet(c.text)])
            for i, (c, score, hl) in enumerate(raw, 1)
        ]

    def _embed_and_knn(
        self, q: str, filters: SearchFilters | None, size: int, timings: dict[str, float]
    ) -> list[tuple[Chunk, float]]:
        t = time.perf_counter()
        vector = self.embedder.embed_queries([q])[0]
        timings["embed"] = _ms(t)
        t = time.perf_counter()
        found = knn_search(self.es, self.index, vector, filters, size)
        timings["knn"] = _ms(t)
        return found

    def _knn(
        self, q: str, filters: SearchFilters | None, k: int, timings: dict[str, float]
    ) -> list[SearchHit]:
        found = self._embed_and_knn(q, filters, k, timings)
        return [
            SearchHit(c, score, i, "knn", [fallback_snippet(c.text)])
            for i, (c, score) in enumerate(found, 1)
        ]

    def _hybrid(
        self, q: str, filters: SearchFilters | None, k: int, timings: dict[str, float], rerank: bool
    ) -> list[SearchHit]:
        pool = max(self.settings.candidate_pool, k)
        t = time.perf_counter()
        lexical = bm25_search(self.es, self.index, q, filters, pool, self.settings)
        timings["bm25"] = _ms(t)
        semantic = self._embed_and_knn(q, filters, pool, timings)

        t = time.perf_counter()
        chunks = {c.chunk_id: c for c, *_ in lexical} | {c.chunk_id: c for c, _ in semantic}
        highlights = {c.chunk_id: hl for c, _, hl in lexical}
        fused = rrf_fuse(
            {"bm25": [c.chunk_id for c, *_ in lexical], "knn": [c.chunk_id for c, _ in semantic]},
            self.settings.rrf_k,
        )
        keep = max(self.settings.rerank_pool, k) if rerank else k
        method = "hybrid"
        fused_hits = [
            SearchHit(
                chunks[cid],
                score,
                i,
                method,
                highlights.get(cid) or [fallback_snippet(chunks[cid].text)],
            )
            for i, (cid, score) in enumerate(fused[:keep], 1)
        ]
        timings["fusion"] = _ms(t)
        if not rerank:
            return fused_hits

        assert self.reranker is not None  # checked in search()
        t = time.perf_counter()
        reranked = self.reranker.rerank(q, fused_hits, k)
        timings["rerank"] = _ms(t)
        return reranked

    # ---- corpus information -----------------------------------------------------------------

    def stats(self) -> dict[str, Any]:
        """{"documents", "chunks", "by_source": {name: {"chunks", "documents"}}, "by_lang": {...}}."""
        doc_card = {"cardinality": {"field": "doc_id", "precision_threshold": 3000}}
        aggs: dict[str, dict[str, Any]] = {
            "docs": doc_card,
            "by_source": {
                "terms": {"field": "source", "size": MAX_AGG_BUCKETS},
                "aggs": {"docs": doc_card},
            },
            "by_lang": {
                "terms": {"field": "lang", "size": MAX_AGG_BUCKETS},
                "aggs": {"docs": doc_card},
            },
        }
        try:
            resp = self.es.search(index=self.index, size=0, track_total_hits=True, aggs=aggs)
        except NotFoundError:
            log.warning("index %s does not exist yet", self.index)
            return {"documents": 0, "chunks": 0, "by_source": {}, "by_lang": {}}

        def buckets(name: str) -> dict[str, dict[str, int]]:
            return {
                b["key"]: {"chunks": b["doc_count"], "documents": b["docs"]["value"]}
                for b in resp["aggregations"][name]["buckets"]
            }

        return {
            "documents": resp["aggregations"]["docs"]["value"],
            "chunks": resp["hits"]["total"]["value"],
            "by_source": buckets("by_source"),
            "by_lang": buckets("by_lang"),
        }

    def list_documents(
        self, source: str | None = None, limit: int = 50, offset: int = 0
    ) -> list[dict[str, Any]]:
        """One row per document (newest first): doc_id, title, source, date, url, lang, doc_type, n_chunks."""
        aggs: dict[str, dict[str, Any]] = {
            "docs": {
                "terms": {
                    "field": "doc_id",
                    "size": min(offset + limit, MAX_AGG_BUCKETS * 10),
                    "order": [{"latest": "desc"}, {"_key": "asc"}],
                },
                "aggs": {
                    "latest": {"max": {"field": "date"}},
                    "meta": {"top_hits": {"size": 1, "_source": {"includes": DOC_SOURCE_FIELDS}}},
                },
            }
        }
        query = {"term": {"source": source}} if source else {"match_all": {}}
        resp = self.es.search(index=self.index, size=0, query=query, aggs=aggs)
        rows = []
        for b in resp["aggregations"]["docs"]["buckets"][offset : offset + limit]:
            meta = b["meta"]["hits"]["hits"][0]["_source"]
            rows.append({**{f: meta.get(f) for f in DOC_SOURCE_FIELDS}, "n_chunks": b["doc_count"]})
        return rows

    def get_chunks(self, ids: list[str]) -> list[Chunk]:
        """Fetch chunks by chunk_id, in the order given; unknown ids are skipped."""
        if not ids:
            return []
        resp = self.es.mget(index=self.index, ids=ids, source_excludes=SOURCE_EXCLUDES)
        return [doc_to_chunk(d["_source"]) for d in resp["docs"] if d.get("found")]


def get_search_service(settings: Settings | None = None) -> SearchService:
    """Production wiring. Model loading is lazy: embedder on first query, reranker on first hybrid_rerank."""
    s = settings or get_settings()
    return SearchService(get_es(s), get_embedder(s), get_reranker(s), s)
