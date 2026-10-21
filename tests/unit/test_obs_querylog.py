from __future__ import annotations

from typing import Any

import pytest
from fastapi import FastAPI, HTTPException, Request
from fastapi.testclient import TestClient

from finsearch.api.middleware import QueryLogMiddleware
from finsearch.api.querylog import QueryLogger, build_doc, query_hash

OBS = {
    "endpoint": "search",
    "method": "hybrid",
    "query": "SNB  Leitzins 2024",
    "n_hits": 7,
    "timings_ms": {"embed": 12.345, "bm25": 5.0, "knn": 6.0, "fusion": 0.5},
    "lang": "de",
    "found": None,
}


class FakeIndices:
    def __init__(self) -> None:
        self.created: list[dict[str, Any]] = []
        self.existing = False

    def exists(self, index: str) -> bool:
        return self.existing

    def create(self, **kwargs: Any) -> None:
        self.created.append(kwargs)
        self.existing = True


class FakeES:
    def __init__(self, fail: bool = False) -> None:
        self.docs: list[dict[str, Any]] = []
        self.indices = FakeIndices()
        self.fail = fail

    def index(self, index: str, document: dict[str, Any]) -> None:
        if self.fail:
            raise ConnectionError("es down")
        self.docs.append({"_index": index, **document})


def make_client(es: FakeES, raw: bool = False) -> tuple[TestClient, QueryLogger]:
    ql = QueryLogger(es, "q-test", log_raw_queries=raw)  # type: ignore[arg-type]
    app = FastAPI()
    app.add_middleware(QueryLogMiddleware, query_logger=ql)

    @app.get("/search")
    def search(request: Request) -> dict[str, str]:
        request.state.obs = OBS
        return {"ok": "1"}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok"}

    @app.get("/static/app.css")
    def static_file() -> str:
        return "body{}"

    @app.get("/boom")
    def boom() -> None:
        raise RuntimeError("kaputt")

    @app.get("/bad")
    def bad() -> None:
        raise HTTPException(status_code=422, detail="nope")

    return TestClient(app, raise_server_exceptions=False), ql


def test_hash_is_normalized_and_stable() -> None:
    assert query_hash("SNB Leitzins") == query_hash("  snb   LEITZINS ")
    assert query_hash("a") != query_hash("b")
    assert len(query_hash("x")) == 16


def test_doc_has_all_fields_and_no_raw_query_by_default() -> None:
    es = FakeES()
    client, ql = make_client(es)
    assert client.get("/search").status_code == 200
    ql.flush()
    (doc,) = es.docs
    assert doc["_index"] == "q-test"
    assert doc["path"] == "/search" and doc["http_method"] == "GET" and doc["status"] == 200
    assert doc["error"] is False and "error_type" not in doc
    assert doc["endpoint"] == "search" and doc["method"] == "hybrid" and doc["lang"] == "de"
    assert doc["query_hash"] == query_hash(OBS["query"]) and doc["query_len"] == 18  # type: ignore[arg-type]
    assert doc["n_hits"] == 7 and doc["zero_hits"] is False
    assert doc["embed_ms"] == 12.35 and doc["fusion_ms"] == 0.5
    assert "rerank_ms" not in doc and "found" not in doc  # unset stays absent
    assert doc["total_ms"] >= 0 and "@timestamp" in doc
    assert "query" not in doc


def test_raw_query_only_with_flag() -> None:
    es = FakeES()
    client, ql = make_client(es, raw=True)
    client.get("/search")
    ql.flush()
    assert es.docs[0]["query"] == OBS["query"]


def test_zero_hits_flag() -> None:
    doc = build_doc(path="/s", http_method="GET", status=200, total_ms=1, obs={"n_hits": 0})
    assert doc["zero_hits"] is True


def test_es_failure_does_not_break_request() -> None:
    client, ql = make_client(FakeES(fail=True))
    assert client.get("/search").status_code == 200
    ql.flush()  # must not raise


def test_unhandled_exception_gives_500_and_error_doc(caplog: pytest.LogCaptureFixture) -> None:
    es = FakeES()
    client, ql = make_client(es)
    assert client.get("/boom").status_code == 500
    ql.flush()
    (doc,) = es.docs
    assert doc["status"] == 500 and doc["error"] is True and doc["error_type"] == "RuntimeError"
    assert any("unhandled error" in r.message for r in caplog.records)


def test_client_error_is_not_an_error() -> None:
    es = FakeES()
    client, ql = make_client(es)
    assert client.get("/bad").status_code == 422
    ql.flush()
    assert es.docs[0]["status"] == 422 and es.docs[0]["error"] is False


def test_health_and_static_are_skipped() -> None:
    es = FakeES()
    client, ql = make_client(es)
    client.get("/health")
    client.get("/static/app.css")
    ql.flush()
    assert es.docs == []


def test_json_log_line_has_no_raw_query(caplog: pytest.LogCaptureFixture) -> None:
    client, ql = make_client(FakeES())
    with caplog.at_level("INFO", logger="finsearch.request"):
        client.get("/search")
    ql.flush()
    line = next(r.message for r in caplog.records if r.name == "finsearch.request")
    assert '"query_hash"' in line and "Leitzins" not in line


def test_ensure_index_is_idempotent_and_survives_es_down() -> None:
    es = FakeES()
    ql = QueryLogger(es, "q-test")  # type: ignore[arg-type]
    assert ql.ensure_index() and ql.ensure_index()
    assert len(es.indices.created) == 1
    assert es.indices.created[0]["mappings"]["properties"]["@timestamp"] == {"type": "date"}

    class Broken:
        class indices:  # noqa: N801
            @staticmethod
            def exists(index: str) -> bool:
                raise ConnectionError("down")

    assert QueryLogger(Broken(), "q").ensure_index() is False  # type: ignore[arg-type]
