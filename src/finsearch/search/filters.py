"""Shared query pieces: SearchFilters -> ES filter clauses, query normalisation, _source selection."""

from __future__ import annotations

from typing import Any

from finsearch.contracts import SearchFilters

MAX_QUERY_CHARS = 1000  # ~150 words; longer input adds no signal and risks clause/token limits
# Never ship vectors or the duplicated analysed text back from ES.
SOURCE_EXCLUDES = ["embedding", "text_de", "text_en"]


def normalize_query(query: str) -> str:
    """Collapse whitespace and truncate. Returns '' for blank input (callers treat that as 'no hits')."""
    return " ".join(query.split())[:MAX_QUERY_CHARS].strip()


def build_filter_clauses(filters: SearchFilters | None) -> list[dict[str, Any]]:
    """Exact-match filters (cached, no scoring) shared by BM25 and kNN."""
    if filters is None:
        return []
    clauses: list[dict[str, Any]] = []
    if filters.sources:
        clauses.append({"terms": {"source": list(filters.sources)}})
    if filters.lang:
        clauses.append({"term": {"lang": filters.lang}})
    if filters.doc_type:
        clauses.append({"term": {"doc_type": filters.doc_type}})
    if filters.date_from or filters.date_to:
        bounds: dict[str, str] = {}
        if filters.date_from:
            bounds["gte"] = filters.date_from.isoformat()
        if filters.date_to:
            bounds["lte"] = filters.date_to.isoformat()
        clauses.append({"range": {"date": bounds}})
    return clauses
