"""The HTTP API on top of the real SearchService, a throwaway ES index and the SYNTHETIC corpus."""

from __future__ import annotations

from collections.abc import Iterator

import pytest
from elasticsearch import Elasticsearch
from fakes import FakeReranker, HashEmbedder
from fastapi.testclient import TestClient

from finsearch.api.deps import Services
from finsearch.api.main import create_app
from finsearch.api.querylog import QueryLogger
from finsearch.config import Settings
from finsearch.search.service import SearchService

pytestmark = pytest.mark.integration

METHODS = ["bm25", "knn", "hybrid", "hybrid_rerank"]


def make_client(
    service: SearchService, settings: Settings, ql: QueryLogger
) -> Iterator[TestClient]:
    cfg = settings.model_copy(update={"api_rate_limit": 0})
    app = create_app(cfg, Services(cfg, search_factory=lambda s: service), ql)
    with TestClient(app) as client:
        yield client


@pytest.fixture(scope="module")
def query_logger(es: Elasticsearch, test_settings: Settings) -> QueryLogger:
    # the unique test prefix makes the session cleanup in tests/conftest.py drop this index too
    return QueryLogger(es, f"{test_settings.es_index_alias}-queries")


@pytest.fixture(scope="module")
def client(
    service: SearchService, test_settings: Settings, query_logger: QueryLogger
) -> Iterator[TestClient]:
    yield from make_client(service, test_settings, query_logger)


@pytest.mark.parametrize("method", METHODS)
def test_search_every_method(client, method):
    r = client.get("/search", params={"q": "Leitzins gesenkt", "method": method, "k": 5})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == method and 1 <= body["n_hits"] <= 5
    hit = body["hits"][0]
    assert hit["rank"] == 1 and hit["method"] == method and hit["highlights"]
    assert hit["chunk_id"] == "snb-speech-0000"
    assert body["timings_ms"]  # the service reports its stage timings
    assert ("rerank" in body["timings_ms"]) == (method == "hybrid_rerank")


def test_search_filters(client):
    hits = client.get("/search", params={"q": "Zinsen", "source": "fed", "k": 20}).json()["hits"]
    assert hits and {h["source"] for h in hits} == {"fed"}
    hits = client.get(
        "/search", params={"q": "Zinsen", "from": "2024-01-01", "to": "2024-12-31", "k": 20}
    ).json()["hits"]
    assert hits and all(h["date"].startswith("2024") for h in hits)
    hits = client.get("/search", params={"q": "Zinsen", "lang": "en", "k": 20}).json()["hits"]
    assert hits and {h["lang"] for h in hits} == {"en"}


def test_search_highlights_are_escaped_for_a_hostile_chunk(client):
    hits = client.get("/search", params={"q": "Inflation outlook", "method": "bm25"}).json()["hits"]
    hostile = next(h for h in hits if h["chunk_id"] == "bank-en-0001")
    assert "<script>" not in "".join(hostile["highlights"])
    assert "&lt;script&gt;" in "".join(hostile["highlights"]) and "<em>" in hostile["highlights"][0]


def test_compare(client):
    body = client.get("/compare", params={"q": "Leitzins Zinsen", "k": 5}).json()
    assert list(body["results"]) == ["bm25", "knn", "hybrid"]
    ids = {m: [h["chunk_id"] for h in r["hits"]] for m, r in body["results"].items()}
    assert all(ids.values())
    assert ids["hybrid"][0] in ids["bm25"] + ids["knn"]  # RRF winner comes from the top of a list


def test_ask_extractive(client):
    r = client.post(
        "/ask", json={"question": "Wie hoch ist der Leitzins der Nationalbank?", "k": 4}
    )
    body = r.json()
    assert r.status_code == 200 and body["found"] and body["provider"] == "extractive"
    ids = {c["chunk_id"] for c in body["used_chunks"]}
    assert body["citations"] and all(c["chunk_id"] in ids for c in body["citations"])
    assert "Leitzins" in body["answer"]


def test_ask_not_found(client):
    body = client.post("/ask", json={"question": "Wie funktioniert Quantenverschränkung?"}).json()
    assert body["found"] is False and body["answer"] == "In den Quellen nicht gefunden"


def test_documents_and_stats(client):
    docs = client.get("/documents", params={"limit": 3}).json()
    assert [d["doc_id"] for d in docs["items"]][0] == "fed-speech"  # newest first (2024-06-12)
    assert len(docs["items"]) == 3 and docs["items"][0]["n_chunks"] == 2
    page2 = client.get("/documents", params={"limit": 3, "offset": 3}).json()["items"]
    assert {d["doc_id"] for d in page2}.isdisjoint({d["doc_id"] for d in docs["items"]})
    only = client.get("/documents", params={"source": "fed"}).json()["items"]
    assert {d["source"] for d in only} == {"fed"} and len(only) == 2
    stats = client.get("/stats").json()
    assert stats["chunks"] == 11 and stats["documents"] == 6
    assert stats["by_source"]["snb"]["chunks"] == 4


def test_health_ok_with_real_elasticsearch(client):
    body = client.get("/health").json()
    assert body["status"] == "ok" and body["elasticsearch"]["chunks"] == 11


def test_requests_land_in_the_query_log(client, query_logger, es):
    client.get("/search", params={"q": "Hypothekarzinsen", "method": "bm25"})
    query_logger.flush()
    es.indices.refresh(index=query_logger.index)
    docs = es.search(index=query_logger.index, query={"term": {"endpoint": "search"}}, size=50)[
        "hits"
    ]["hits"]
    doc = next(d["_source"] for d in docs if d["_source"].get("method") == "bm25")
    assert doc["path"] == "/search" and doc["status"] == 200 and doc["n_hits"] >= 1
    assert (
        "query_hash" in doc and "bm25_ms" in doc and "query" not in doc
    )  # raw text off by default


def test_missing_index_is_a_503_with_hint_not_a_traceback(es, test_settings, query_logger):
    empty = SearchService(
        es, HashEmbedder(), FakeReranker(), test_settings, index="finsearch-test-none"
    )
    for c in make_client(empty, test_settings, query_logger):
        r = c.get("/search", params={"q": "Zins"})
        assert r.status_code == 503 and r.json()["error"] == "index_missing"
        assert "make ingest" in r.json()["hint"]
        health = c.get("/health").json()
        assert health["status"] == "degraded" and "make ingest" in health["hint"]
        assert c.get("/stats").json()["chunks"] == 0  # the service reports an empty index itself


def test_elasticsearch_down_is_a_503_and_health_stays_200(test_settings, query_logger):
    dead = Elasticsearch("http://127.0.0.1:1", max_retries=0, request_timeout=1)
    svc = SearchService(dead, HashEmbedder(), FakeReranker(), test_settings, index="x")
    for c in make_client(svc, test_settings, query_logger):
        assert c.get("/search", params={"q": "Zins"}).status_code == 503
        r = c.get("/health")
        assert r.status_code == 200 and r.json()["status"] == "degraded"
