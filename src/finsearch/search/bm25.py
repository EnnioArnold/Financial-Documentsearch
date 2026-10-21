"""Lexical search: multi_match over the language-analysed text fields."""

from __future__ import annotations

from typing import Any

from elasticsearch import Elasticsearch

from finsearch.config import Settings, get_settings
from finsearch.contracts import Chunk, SearchFilters
from finsearch.search.filters import SOURCE_EXCLUDES, build_filter_clauses, normalize_query
from finsearch.search.mapping import doc_to_chunk

HIGHLIGHT_FRAGMENT_CHARS = 200
HIGHLIGHT_FRAGMENTS = 2
# best_fields: the same text is analysed three ways, so the *best* analyser should decide (not the sum,
# which double-counts every term). tie_breaker lets the other fields add a little. See docs/DECISIONS.md.
MULTI_MATCH_TYPE = "best_fields"
TIE_BREAKER = 0.3

# (chunk, score, highlight snippets) in rank order
RawHit = tuple[Chunk, float, list[str]]


def build_bm25_request(
    query: str, filters: SearchFilters | None, size: int, settings: Settings | None = None
) -> dict[str, Any]:
    """Keyword arguments for `es.search` (note: `source` is the client's name for `_source`)."""
    s = settings or get_settings()
    must = {
        "multi_match": {
            "query": query,
            "type": MULTI_MATCH_TYPE,
            "tie_breaker": TIE_BREAKER,
            "fields": [f"text_de^{s.boost_de}", f"text_en^{s.boost_en}", f"text^{s.boost_text}"],
        }
    }
    return {
        "query": {"bool": {"must": [must], "filter": build_filter_clauses(filters)}},
        "size": size,
        "source": {"excludes": SOURCE_EXCLUDES},
        "highlight": {
            "fields": {
                field: {
                    "fragment_size": HIGHLIGHT_FRAGMENT_CHARS,
                    "number_of_fragments": HIGHLIGHT_FRAGMENTS,
                }
                for field in ("text_de", "text_en", "text")
            },
            "encoder": "html",  # escape the text, keep only our <em> tags => safe to render
        },
    }


def pick_highlights(chunk: Chunk, highlight: dict[str, list[str]]) -> list[str]:
    """Take the fragments of the field matching the chunk's language first."""
    order = ("text_de", "text_en", "text") if chunk.lang == "de" else ("text_en", "text_de", "text")
    for field in order:
        if highlight.get(field):
            return list(highlight[field])
    return []


def bm25_search(
    es: Elasticsearch,
    index: str,
    query: str,
    filters: SearchFilters | None,
    size: int,
    settings: Settings | None = None,
) -> list[RawHit]:
    """BM25 top-`size`. Blank queries return [] without calling ES."""
    q = normalize_query(query)
    if not q:
        return []
    resp = es.search(index=index, **build_bm25_request(q, filters, size, settings))
    hits: list[RawHit] = []
    for h in resp["hits"]["hits"]:
        chunk = doc_to_chunk(h["_source"])
        hits.append((chunk, float(h["_score"]), pick_highlights(chunk, h.get("highlight", {}))))
    return hits
