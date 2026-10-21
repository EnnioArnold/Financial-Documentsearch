"""Every endpoint with a fake SearchService (no Elasticsearch)."""

from __future__ import annotations

import pytest
from api_fakes import XSS, EsConnectionError, FakeLLM, make_env, not_found_error
from fastapi.testclient import TestClient
from rag_fakes import settings

from finsearch.api.deps import Services, percentile
from finsearch.api.main import create_app


@pytest.fixture
def env():
    e = make_env()
    with e.client:  # runs the lifespan (ensure_index / close)
        yield e


def test_lifespan_ensures_and_closes_query_log():
    e = make_env()
    with e.client:
        assert e.logger.ensured and not e.logger.closed
    assert e.logger.closed


@pytest.mark.parametrize("method", ["bm25", "knn", "hybrid", "hybrid_rerank"])
def test_search_all_methods(env, method):
    r = env.client.get("/search", params={"q": "Leitzins", "method": method, "k": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["method"] == method and body["n_hits"] == 2 and body["k"] == 2
    hit = body["hits"][0]
    assert {"chunk_id", "doc_id", "title", "source", "date", "page", "url", "score", "rank"} <= set(
        hit
    )
    assert hit["method"] == method and hit["rank"] == 1 and hit["date"] == "2024-06-20"
    assert hit["snippet"] and hit["text"] and hit["highlights"]
    assert body["timings_ms"]["bm25"] == 1.5
    assert env.search.calls[-1][:3] == ("Leitzins", method, 2)


def test_search_defaults_and_filters_are_forwarded(env):
    r = env.client.get(
        "/search",
        params={
            "q": "  Zins  ",
            "source": "snb",
            "from": "2024-01-01",
            "to": "2024-12-31",
            "lang": "de",
            "doc_type": "speech",
        },
    )
    assert r.status_code == 200
    q, method, k, filters = env.search.calls[-1]
    assert (q, method, k) == ("Zins", "hybrid", 10)  # stripped; defaults
    assert filters.sources == ["snb"] and filters.lang == "de" and filters.doc_type == "speech"
    assert (
        filters.date_from.isoformat() == "2024-01-01"
        and filters.date_to.isoformat() == "2024-12-31"
    )


def test_search_blank_filter_values_mean_unset(env):
    r = env.client.get("/search", params={"q": "x", "source": "", "lang": "", "from": "", "to": ""})
    assert r.status_code == 200
    assert env.search.calls[-1][3].sources == []


def test_compare_returns_three_methods_side_by_side(env):
    r = env.client.get("/compare", params={"q": "Leitzins", "k": 3})
    assert r.status_code == 200
    body = r.json()
    assert list(body["results"]) == ["bm25", "knn", "hybrid"]
    assert all(
        body["results"][m]["method"] == m and body["results"][m]["n_hits"] == 3
        for m in body["results"]
    )
    assert body["results"]["hybrid"]["total_ms"] == 9.0


def test_ask_extractive_answer_with_citations(env):
    r = env.client.post(
        "/ask", json={"question": "Wie hoch ist der Leitzins im Juni 2024?", "k": 2}
    )
    assert r.status_code == 200
    body = r.json()
    assert body["found"] is True and body["provider"] == "extractive"
    assert body["citations"][0]["n"] == 1 and "[1]" in body["answer"]
    assert body["used_chunks"][0]["chunk_id"] == "doc1-0000"
    assert body["dropped_citations"] == [] and body["confidence_note"]


def test_ask_not_found_state(env):
    r = env.client.post("/ask", json={"question": "Quantenverschränkung bei Kryptowährungen?"})
    body = r.json()
    assert r.status_code == 200 and body["found"] is False
    assert body["answer"] == "In den Quellen nicht gefunden" and body["citations"] == []


def test_ask_uses_configured_llm_and_forwards_filters():
    llm = FakeLLM("Die SNB senkte den Leitzins auf 1,25 Prozent [1].", name="fake")
    e = make_env(llm=llm)
    r = e.client.post(
        "/ask",
        json={
            "question": "Wie hoch ist der Leitzins im Juni 2024?",
            "method": "bm25",
            "filters": {"source": "snb", "from": "2024-01-01"},
        },
    )
    body = r.json()
    assert body["provider"] == "fake" and body["found"] and "[1]" in body["answer"]
    _, method, k, filters = e.search.calls[-1]
    assert method == "bm25" and k == 6 and filters.sources == ["snb"]


def test_documents_paged(env):
    r = env.client.get("/documents", params={"limit": 5, "source": "snb"})
    body = r.json()
    assert r.status_code == 200 and body["limit"] == 5 and body["offset"] == 0
    assert body["items"][0]["doc_id"] == "doc1" and body["items"][0]["n_chunks"] == 2
    assert env.client.get("/documents", params={"offset": 1}).json()["items"] == []


def test_stats(env):
    body = env.client.get("/stats").json()
    assert body["chunks"] == 3 and "snb" in body["by_source"]


def test_runtime_stats_counts_and_percentiles(env):
    for _ in range(3):
        env.client.get("/search", params={"q": "Leitzins", "method": "bm25"})
    env.client.get("/search", params={"q": ""})  # 422, not an error of ours
    env.search.error = EsConnectionError("down")
    env.client.get("/search", params={"q": "x", "method": "knn"})  # 503 -> counted as error
    rows = {
        (r["endpoint"], r["method"]): r
        for r in env.client.get("/stats/runtime").json()["endpoints"]
    }
    assert rows[("search", "bm25")]["count"] == 3 and rows[("search", "bm25")]["errors"] == 0
    assert rows[("search", "knn")]["errors"] == 1  # the 503 is labelled and counted as an error
    assert rows[("/search", "GET")]["count"] == 1  # the 422 never reached the handler: unlabelled
    assert rows[("/search", "GET")]["errors"] == 0
    assert rows[("search", "bm25")]["p95_ms"] >= rows[("search", "bm25")]["p50_ms"] >= 0
    assert env.client.get("/stats/runtime").json()["since"]


def test_runtime_stats_percentile_helper():
    assert percentile([], 50) == 0.0
    assert percentile([1, 2, 3, 4], 50) == 2 and percentile([1, 2, 3, 4], 95) == 4


def test_health_ok(env):
    body = env.client.get("/health").json()
    assert body["status"] == "ok" and body["elasticsearch"] == {
        "reachable": True,
        "index": "fake-index",
        "chunks": 3,
    }
    assert body["llm"]["provider"] == "extractive" and body["llm"]["fallback"] is False


def test_health_degraded_when_es_down(env):
    env.search.es.error = EsConnectionError("down")
    r = env.client.get("/health")
    body = r.json()
    assert r.status_code == 200 and body["status"] == "degraded"
    assert body["elasticsearch"]["reachable"] is False and "make up" in body["hint"]


def test_health_degraded_when_index_missing_or_empty(env):
    env.search.es.error = not_found_error()
    body = env.client.get("/health").json()
    assert body["status"] == "degraded" and body["elasticsearch"]["reachable"] is True
    assert "make ingest" in body["hint"]
    env.search.es.error, env.search.es._count = None, 0
    body = env.client.get("/health").json()
    assert body["status"] == "degraded" and "make ingest" in body["hint"]


def test_health_survives_broken_service_factory():
    def boom(_):
        raise RuntimeError("cannot build")

    cfg = settings()
    client = TestClient(create_app(cfg, Services(cfg, search_factory=boom), _Logger()))
    body = client.get("/health").json()
    assert body["status"] == "degraded" and body["elasticsearch"]["chunks"] == 0


class _Logger:
    def ensure_index(self):
        return True

    def record(self, **_):
        pass

    def close(self):
        pass


def test_es_down_gives_503_with_hint_on_every_data_endpoint(env):
    env.search.error = EsConnectionError("down")
    calls = [
        env.client.get("/search", params={"q": "x"}),
        env.client.get("/compare", params={"q": "x"}),
        env.client.post("/ask", json={"question": "Leitzins im Juni?"}),
        env.client.get("/documents"),
        env.client.get("/stats"),
    ]
    for r in calls:
        assert r.status_code == 503, r.request.url
        body = r.json()
        assert body["error"] == "search_backend_unavailable" and "make up" in body["hint"]
        assert "Traceback" not in r.text


def test_missing_index_gives_503_index_empty_hint(env):
    env.search.error = not_found_error()
    r = env.client.get("/search", params={"q": "x"})
    assert r.status_code == 503
    assert r.json()["error"] == "index_missing" and "make ingest" in r.json()["hint"]


def test_rerank_unavailable_gives_503_not_a_traceback(env):
    env.search.rerank_error = OSError("model cannot be downloaded")
    for r in (
        env.client.get("/search", params={"q": "x", "method": "hybrid_rerank"}),
        env.client.post("/ask", json={"question": "Leitzins im Juni?", "method": "hybrid_rerank"}),
    ):
        assert r.status_code == 503
        body = r.json()
        assert body["error"] == "reranker_unavailable" and "Hybrid" in body["hint"]
        assert "OSError" not in r.text and "Traceback" not in r.text
    assert env.client.get("/search", params={"q": "x", "method": "hybrid"}).status_code == 200


def test_non_rerank_unexpected_error_is_a_clean_500(env):
    env.search.error = RuntimeError("boom secret detail")
    r = env.client.get("/search", params={"q": "x", "method": "hybrid"})
    assert r.status_code == 500 and r.json()["error"] == "internal_error"
    assert "secret" not in r.text and "Traceback" not in r.text


def test_llm_fallback_when_provider_not_configured():
    e = make_env(llm_provider="gemini", gemini_api_key="")  # real factory raises LLMError
    with e.client:
        health = e.client.get("/health").json()["llm"]
        assert health["provider"] == "extractive" and health["requested"] == "gemini"
        assert health["fallback"] is True and "API-Key" in health["note"]
        body = e.client.post(
            "/ask", json={"question": "Wie hoch ist der Leitzins im Juni 2024?"}
        ).json()
    assert body["found"] and body["provider"] == "extractive"
    assert "gemini" in body["confidence_note"] and "extraktive" in body["confidence_note"]


def test_llm_fallback_for_unknown_provider():
    e = make_env(llm_provider="doesnotexist")
    assert e.client.get("/health").json()["llm"]["fallback"] is True


def test_ui_pages_render_german_ui_with_examples_and_footer(env):
    for path, marker in [
        ("/", "Dokumente durchsuchen"),
        ("/frage", "Frage stellen"),
        ("/vergleich", "Methoden im Vergleich"),
    ]:
        r = env.client.get(path)
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/html")
        assert (
            marker in r.text
            and "Beispiel-Fragen" in r.text
            and "What did the Federal Reserve" in r.text
        )
        assert (
            "Lokale Demo – Sicherheit von Elasticsearch ist deaktiviert, nicht für Produktion"
            in r.text
        )
        assert "<script>" not in r.text and "http://" not in r.text.replace("http://www.w3.org", "")
        assert '<script src="/static/app.js" defer></script>' in r.text


def test_static_assets_and_no_docs_cdn(env):
    assert env.client.get("/static/app.js").status_code == 200
    assert env.client.get("/static/style.css").status_code == 200
    assert env.client.get("/docs").status_code == 404
    assert "/search" in env.client.get("/openapi.json").json()["paths"]


def test_jinja_autoescape_is_on(env):
    jinja = env.app.state.templates.env
    assert jinja.autoescape("page.html")
    assert jinja.from_string("{{ x }}").render(x=XSS) == "&lt;script&gt;alert(1)&lt;/script&gt;"


def test_runtime_stats_key_count_is_bounded():
    from finsearch.api.deps import MAX_STAT_KEYS, RuntimeStats

    stats = RuntimeStats()
    for i in range(MAX_STAT_KEYS * 3):
        stats.record(f"/probe-{i}", "GET", 404, 1.0)
    assert len(stats.snapshot()["endpoints"]) <= MAX_STAT_KEYS + 1
