"""Search against the running Elasticsearch with a throwaway index and the SYNTHETIC corpus."""

import json
import uuid
from dataclasses import asdict
from datetime import date

import pytest
from fakes import HashEmbedder
from synthetic_corpus import CORPUS

from finsearch.contracts import SearchFilters
from finsearch.search.indexer import index_chunks, index_chunks_jsonl
from finsearch.search.mapping import create_index, switch_alias

pytestmark = pytest.mark.integration

METHODS = ["bm25", "knn", "hybrid", "hybrid_rerank"]


def ids(result):
    return [h.chunk.chunk_id for h in result.hits]


# ---- filters ----------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    ("filters", "check"),
    [
        (SearchFilters(sources=["fed"]), lambda c: c.source == "fed"),
        (SearchFilters(sources=["snb", "other"]), lambda c: c.source in ("snb", "other")),
        (SearchFilters(lang="de"), lambda c: c.lang == "de"),
        (SearchFilters(doc_type="annual_report"), lambda c: c.doc_type == "annual_report"),
        (
            SearchFilters(date_from=date(2024, 1, 1), date_to=date(2024, 12, 31)),
            lambda c: c.date is not None and c.date.year == 2024,
        ),
        (
            SearchFilters(date_to=date(2022, 12, 31)),
            lambda c: c.date is not None and c.date.year <= 2022,
        ),
        (
            SearchFilters(sources=["snb"], lang="de", doc_type="speech"),
            lambda c: c.doc_id == "snb-speech",
        ),
    ],
)
def test_filters_are_applied(service, method, filters, check):
    res = service.search("interest rate inflation Zins Bilanz", method, k=20, filters=filters)
    assert res.hits, "filter should still leave matching documents"
    assert all(check(h.chunk) for h in res.hits)


def test_filter_without_matches_returns_empty(service):
    res = service.search("Inflation", "hybrid", filters=SearchFilters(sources=["does-not-exist"]))
    assert res.hits == []


# ---- lexical behaviour ------------------------------------------------------------------------


def test_german_inflection_plural_finds_singular(service, es, corpus_index):
    # "Zinssätze" (plural) must find the chunk that only contains "Zinssatz" (singular)
    assert "snb-annual-0001" in ids(service.search("Zinssätze", "bm25"))
    # ...and that is the German analyzer's doing: the plain `text` field cannot match it
    plain = es.search(index=corpus_index, query={"match": {"text": "Zinssätze"}})
    assert plain["hits"]["hits"] == []


def test_german_noun_inflection_banken_finds_bank(service):
    res = service.search("Bank", "bm25", filters=SearchFilters(lang="de"))
    assert "snb-annual-0001" in ids(res)  # text contains "Banken"


def test_english_inflection(service):
    assert "fed-minutes-0000" in ids(
        service.search("raising rates", "bm25")
    )  # text: "raised ... rate"


def test_bm25_highlight_present_and_escaped(service):
    res = service.search("Leitzins", "bm25")
    assert res.hits[0].chunk.chunk_id == "snb-speech-0000"
    assert "<em>Leitzins</em>" in res.hits[0].highlights[0]
    risky = service.search("alert Inflation outlook", "bm25")
    snippet = " ".join(risky.hits[0].highlights)
    assert "<script>" not in snippet and "&lt;script&gt;" in snippet


def test_knn_and_hybrid_hits_always_have_a_snippet(service):
    for method in ("knn", "hybrid", "hybrid_rerank"):
        assert all(h.highlights for h in service.search("Treasury securities", method).hits)


# ---- robustness -------------------------------------------------------------------------------


@pytest.mark.parametrize("method", METHODS)
@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        '"AND OR NOT" ( [ { \\ / ~ * ? : ^ ! + - && ||',
        "💥 \x00 <script>",
        "Zins " * 3000,
        "a" * 20_000,
    ],
)
def test_odd_queries_never_crash(service, method, query):
    res = service.search(query, method)
    assert res.query == query and isinstance(res.hits, list)


def test_scores_and_ranks_are_consistent(service):
    for method in METHODS:
        res = service.search("Inflation", method, k=5)
        assert [h.rank for h in res.hits] == list(range(1, len(res.hits) + 1))
        assert all(h.method == method for h in res.hits)
        assert res.total_ms >= sum(res.timings_ms.values()) * 0.9


def test_hybrid_scores_are_rrf_scores(service):
    res = service.search("Inflation", "hybrid", k=3)
    assert all(0 < h.score <= 2 / 61 + 1e-9 for h in res.hits)


# ---- corpus information -----------------------------------------------------------------------


def test_stats(service):
    st = service.stats()
    assert st["chunks"] == len(CORPUS) and st["documents"] == len({c.doc_id for c in CORPUS})
    assert st["by_source"]["snb"] == {"chunks": 4, "documents": 2}
    assert set(st["by_lang"]) == {"de", "en"}


def test_list_documents(service):
    docs = service.list_documents()
    assert len(docs) == 6
    assert {"doc_id", "title", "source", "date", "url", "lang", "doc_type", "n_chunks"} <= set(
        docs[0]
    )
    assert docs[0]["doc_id"] == "fed-speech"  # newest dated document first
    assert docs[-1]["doc_id"] == "bank-en"  # undated last
    assert [d["doc_id"] for d in service.list_documents(source="snb")] == [
        "snb-speech",
        "snb-annual",
    ]
    assert [d["doc_id"] for d in service.list_documents(limit=2, offset=1)] == [
        "snb-speech",
        "bank-de",
    ]
    assert next(d for d in docs if d["doc_id"] == "snb-annual")["n_chunks"] == 2


def test_get_chunks_keeps_order_and_skips_unknown(service):
    got = service.get_chunks(["fed-speech-0001", "nope", "snb-speech-0000"])
    assert [c.chunk_id for c in got] == ["fed-speech-0001", "snb-speech-0000"]
    assert got[1].date == date(2024, 3, 21) and "Leitzins" in got[1].text


# ---- indexing & aliases -----------------------------------------------------------------------


def test_indexing_same_file_twice_is_idempotent(es, test_settings, index_name, tmp_path):
    path = tmp_path / "chunks.jsonl"
    rows = [{**asdict(c), "date": c.date.isoformat() if c.date else None} for c in CORPUS]
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n{broken\n", encoding="utf-8")
    first = index_chunks_jsonl(path, test_settings, HashEmbedder(), index_name, es)
    assert first["indexed"] == len(CORPUS) and first["failed"] == 0 and first["took_s"] >= 0
    assert es.count(index=index_name)["count"] == len(CORPUS)
    index_chunks_jsonl(path, test_settings, HashEmbedder(), index_name, es)
    assert es.count(index=index_name)["count"] == len(CORPUS)


def test_bad_document_is_counted_as_failed_not_raised(es, test_settings, index_name):
    # strict mapping: a malformed vector size is rejected by ES for that document only
    class WrongDim(HashEmbedder):
        def embed_passages(self, texts):
            return [[0.1] * 3 for _ in texts]

    stats = index_chunks(CORPUS[:2], test_settings, WrongDim(), index_name, es)
    assert stats["failed"] == 2 and stats["indexed"] == 0


def test_create_index_is_idempotent_and_alias_switch_is_atomic(es, test_settings):
    alias = f"{test_settings.es_index_alias}-al{uuid.uuid4().hex[:6]}"
    v1, v2 = f"{alias}-v1", f"{alias}-v2"
    assert create_index(es, test_settings, index_name=v1) == v1
    assert create_index(es, test_settings, index_name=v1) == v1  # second call: no error
    create_index(es, test_settings, index_name=v2)
    index_chunks(CORPUS[:2], test_settings, HashEmbedder(), v1, es)
    switch_alias(es, alias, v1)
    assert list(es.indices.get_alias(name=alias)) == [v1]
    assert es.count(index=alias)["count"] == 2
    switch_alias(
        es, alias, v2
    )  # re-index: new index becomes live, old one is detached in the same call
    assert list(es.indices.get_alias(name=alias)) == [v2]
    assert es.count(index=alias)["count"] == 0
