from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from elasticsearch import Elasticsearch
from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from finsearch.api.middleware import QueryLogMiddleware
from finsearch.api.querylog import QueryLogger
from finsearch.config import get_settings

pytestmark = pytest.mark.integration


@pytest.fixture
def es() -> Iterator[Elasticsearch]:
    client = Elasticsearch(get_settings().es_url)
    yield client
    client.close()


def test_request_is_written_and_readable(es: Elasticsearch) -> None:
    index = f"finsearch-queries-test-{uuid.uuid4().hex[:8]}"
    ql = QueryLogger(es, index, log_raw_queries=True)
    try:
        assert ql.ensure_index() and ql.ensure_index()  # idempotent
        app = FastAPI()
        app.add_middleware(QueryLogMiddleware, query_logger=ql)

        @app.get("/search")
        def search(request: Request) -> dict[str, bool]:
            request.state.obs = {
                "endpoint": "search",
                "method": "bm25",
                "query": "Zins",
                "n_hits": 0,
                "timings_ms": {"bm25": 3.0},
            }
            return {"ok": True}

        assert TestClient(app).get("/search").status_code == 200
        ql.flush()
        es.indices.refresh(index=index)
        hits = es.search(index=index, query={"term": {"zero_hits": True}})["hits"]["hits"]
        assert len(hits) == 1
        doc = hits[0]["_source"]
        assert doc["method"] == "bm25" and doc["query"] == "Zins" and doc["bm25_ms"] == 3.0
        mapping = es.indices.get_mapping(index=index)[index]["mappings"]["properties"]
        assert mapping["total_ms"]["type"] == "float" and mapping["@timestamp"]["type"] == "date"
    finally:
        ql.close()
        es.indices.delete(index=index, ignore_unavailable=True)
