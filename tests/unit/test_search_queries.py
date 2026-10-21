"""Pure query-building / parsing helpers (no Elasticsearch)."""

import json
from datetime import date

import pytest
from fakes import HashEmbedder, make_chunk

from finsearch.config import Settings
from finsearch.contracts import SearchFilters
from finsearch.search.bm25 import build_bm25_request, pick_highlights
from finsearch.search.filters import MAX_QUERY_CHARS, build_filter_clauses, normalize_query
from finsearch.search.indexer import load_chunks
from finsearch.search.service import fallback_snippet
from finsearch.search.vector import MIN_NUM_CANDIDATES, build_knn_request


def test_normalize_query():
    assert normalize_query("  Leitzins \n\t SNB  ") == "Leitzins SNB"
    assert normalize_query("   \n\t ") == ""
    assert len(normalize_query("wort " * 5000)) <= MAX_QUERY_CHARS


def test_filter_clauses_none_and_empty():
    assert build_filter_clauses(None) == []
    assert build_filter_clauses(SearchFilters()) == []


def test_filter_clauses_all():
    f = SearchFilters(
        sources=["snb", "fed"],
        lang="de",
        doc_type="speech",
        date_from=date(2024, 1, 1),
        date_to=date(2024, 12, 31),
    )
    assert build_filter_clauses(f) == [
        {"terms": {"source": ["snb", "fed"]}},
        {"term": {"lang": "de"}},
        {"term": {"doc_type": "speech"}},
        {"range": {"date": {"gte": "2024-01-01", "lte": "2024-12-31"}}},
    ]


def test_filter_clauses_open_ended_date_range():
    assert build_filter_clauses(SearchFilters(date_from=date(2024, 1, 1))) == [
        {"range": {"date": {"gte": "2024-01-01"}}}
    ]
    assert build_filter_clauses(SearchFilters(date_to=date(2024, 1, 1))) == [
        {"range": {"date": {"lte": "2024-01-01"}}}
    ]


def test_bm25_request_uses_multi_match_with_boosts_and_filters():
    s = Settings(boost_de=1.0, boost_en=0.8, boost_text=0.5)
    req = build_bm25_request('Zins "AND" (OR) [x] ~*?', SearchFilters(lang="de"), 7, s)
    mm = req["query"]["bool"]["must"][0]["multi_match"]
    assert mm["query"] == 'Zins "AND" (OR) [x] ~*?'  # passed through as data, not query syntax
    assert mm["fields"] == ["text_de^1.0", "text_en^0.8", "text^0.5"]
    assert req["query"]["bool"]["filter"] == [{"term": {"lang": "de"}}]
    assert req["size"] == 7
    assert "query_string" not in json.dumps(req)
    assert req["highlight"]["encoder"] == "html"
    assert "embedding" in req["source"]["excludes"]


def test_knn_request_filter_and_candidates():
    req = build_knn_request([0.1, 0.2], SearchFilters(sources=["snb"]), k=50)
    knn = req["knn"]
    assert knn["k"] == 50 and knn["num_candidates"] == 150 and req["size"] == 50
    assert knn["filter"] == {"bool": {"filter": [{"terms": {"source": ["snb"]}}]}}
    small = build_knn_request([0.1], None, k=3)
    assert small["knn"]["num_candidates"] == MIN_NUM_CANDIDATES
    assert "filter" not in small["knn"]


def test_pick_highlights_prefers_chunk_language():
    hl = {"text_de": ["<em>Zins</em>"], "text_en": ["<em>x</em>"]}
    assert pick_highlights(make_chunk(lang="de"), hl) == ["<em>Zins</em>"]
    assert pick_highlights(make_chunk(lang="en"), hl) == ["<em>x</em>"]
    assert pick_highlights(make_chunk(lang="en"), {}) == []


def test_fallback_snippet_truncates_and_escapes():
    long = "wort " * 100
    snippet = fallback_snippet(long, 50)
    assert len(snippet) <= 52 and snippet.endswith("…")
    assert fallback_snippet("a <b> & c") == "a &lt;b&gt; &amp; c"


def test_load_chunks_is_tolerant(tmp_path):
    good = {
        "chunk_id": "d-0000",
        "doc_id": "d",
        "ord": 0,
        "text": "Hallo Welt",
        "page": 2,
        "source": "snb",
        "title": "T",
        "url": "u",
        "lang": "de",
        "doc_type": "speech",
        "date": "2024-03-21",
        "extra": 1,
    }
    nodate = {**good, "chunk_id": "d-0001", "ord": 1, "date": None, "lang": "fr"}
    lines = [
        json.dumps(good),
        "",
        "{not json",
        json.dumps({"chunk_id": "x"}),
        json.dumps({**good, "chunk_id": "d-0002", "text": "   "}),
        json.dumps(nodate),
        json.dumps({**good, "chunk_id": "d-0003", "date": "2024-03-21T10:00:00"}),
    ]
    path = tmp_path / "chunks.jsonl"
    path.write_text("\n".join(lines), encoding="utf-8")
    chunks = list(load_chunks(path))
    assert [c.chunk_id for c in chunks] == ["d-0000", "d-0001", "d-0003"]
    assert chunks[0].date == date(2024, 3, 21) and chunks[0].page == 2
    assert chunks[1].date is None and chunks[1].lang == "other"
    assert chunks[2].date == date(2024, 3, 21)


def test_hash_embedder_is_normalised_deterministic_and_lexical():
    e = HashEmbedder()
    a, b, c = e.embed_queries(["Leitzins der SNB", "Leitzins SNB", "Gold Bilanz"])
    assert len(a) == e.dim == 384
    assert sum(x * x for x in a) == pytest.approx(1.0)
    assert e.embed_queries(["Leitzins der SNB"])[0] == a
    dot = lambda u, v: sum(x * y for x, y in zip(u, v, strict=True))  # noqa: E731
    assert dot(a, b) > dot(a, c)
    assert sum(x * x for x in e.embed_queries([""])[0]) == pytest.approx(1.0)
