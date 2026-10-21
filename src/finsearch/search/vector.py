"""Dense retrieval: approximate kNN over the embedding field, filters applied inside the search."""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch

from finsearch.contracts import Chunk, SearchFilters
from finsearch.search.filters import SOURCE_EXCLUDES, build_filter_clauses
from finsearch.search.mapping import doc_to_chunk

NUM_CANDIDATES_FACTOR = 3  # HNSW candidates per shard = 3 x k: recall/latency trade-off
MIN_NUM_CANDIDATES = 50
MAX_NUM_CANDIDATES = 10_000  # ES hard limit


def build_knn_request(vector: list[float], filters: SearchFilters | None, k: int) -> dict[str, Any]:
    knn: dict[str, Any] = {
        "field": "embedding",
        "query_vector": vector,
        "k": k,
        "num_candidates": min(
            max(k * NUM_CANDIDATES_FACTOR, MIN_NUM_CANDIDATES), MAX_NUM_CANDIDATES
        ),
    }
    clauses = build_filter_clauses(filters)
    if clauses:
        knn["filter"] = {"bool": {"filter": clauses}}  # pre-filter: kNN still returns k hits
    return {"knn": knn, "size": k, "source": {"excludes": SOURCE_EXCLUDES}}


def knn_search(
    es: Elasticsearch, index: str, vector: list[float], filters: SearchFilters | None, k: int
) -> list[tuple[Chunk, float]]:
    resp = es.search(index=index, **build_knn_request(vector, filters, k))
    return [(doc_to_chunk(h["_source"]), float(h["_score"])) for h in resp["hits"]["hits"]]
