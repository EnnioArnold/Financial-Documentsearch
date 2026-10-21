"""Rate limit, CORS, security headers, XSS, error pages and query-log hand-off."""

from __future__ import annotations

import pytest
from api_fakes import CHUNKS, XSS, EsConnectionError, make_env
from fastapi.testclient import TestClient
from rag_fakes import settings

from finsearch.api.deps import ApiError, RateLimiter
from finsearch.api.main import SECURITY_HEADERS, app, create_app, parse_origins
from finsearch.api.middleware import QueryLogMiddleware
from finsearch.api.schemas import HitOut
from finsearch.config import ROOT
from finsearch.contracts import SearchHit

ALLOWED = "http://127.0.0.1:8000"
HTML = {"accept": "text/html,application/xhtml+xml"}


# ---- rate limit ----------------------------------------------------------------------------


def test_rate_limit_returns_429_with_retry_after_on_search_ask_compare():
    e = make_env(api_rate_limit=3, api_rate_window_s=60)
    for _ in range(3):
        assert e.client.get("/search", params={"q": "x"}).status_code == 200
    r = e.client.get("/search", params={"q": "x"})
    assert r.status_code == 429 and r.json()["error"] == "rate_limited"
    assert 1 <= int(r.headers["retry-after"]) <= 60
    assert e.client.get("/compare", params={"q": "x"}).status_code == 429  # one budget per client
    assert e.client.post("/ask", json={"question": "Leitzins im Juni?"}).status_code == 429
    for path in ("/health", "/stats", "/documents", "/", "/stats/runtime"):
        assert e.client.get(path).status_code == 200, path  # not limited


def test_rate_limit_zero_switches_it_off():
    e = make_env(api_rate_limit=0)
    assert all(e.client.get("/search", params={"q": "x"}).status_code == 200 for _ in range(20))


def test_rate_limiter_window_resets_and_clients_are_independent():
    now = [0.0]
    rl = RateLimiter(2, 10, clock=lambda: now[0])
    assert rl.check("a") is None and rl.check("a") is None
    assert rl.check("a") == 10
    assert rl.check("b") is None  # other client
    now[0] = 4.0
    assert rl.check("a") == 6
    now[0] = 10.0
    assert rl.check("a") is None  # new window


def test_rate_limiter_memory_is_bounded(monkeypatch):
    import finsearch.api.deps as deps

    monkeypatch.setattr(deps, "MAX_TRACKED_CLIENTS", 5)
    now = [0.0]
    rl = RateLimiter(1, 10, clock=lambda: now[0])
    for i in range(5):
        rl.check(f"old{i}")
    now[0] = 11.0  # all old windows expired
    rl.check("new")
    assert list(rl._windows) == ["new"]


# ---- CORS ----------------------------------------------------------------------------------


def test_cors_allows_only_configured_origins():
    c = make_env().client
    ok = c.get("/health", headers={"origin": ALLOWED})
    assert ok.headers["access-control-allow-origin"] == ALLOWED
    local = c.get("/health", headers={"origin": "http://localhost:8000"})
    assert local.headers["access-control-allow-origin"] == "http://localhost:8000"
    evil = c.get("/health", headers={"origin": "https://evil.example"})
    assert "access-control-allow-origin" not in evil.headers


def test_cors_preflight():
    c = make_env().client
    pre = c.options(
        "/ask",
        headers={
            "origin": ALLOWED,
            "access-control-request-method": "POST",
            "access-control-request-headers": "content-type",
        },
    )
    assert pre.status_code == 200 and pre.headers["access-control-allow-origin"] == ALLOWED
    assert "DELETE" not in pre.headers["access-control-allow-methods"]
    bad = c.options(
        "/ask", headers={"origin": "https://evil.example", "access-control-request-method": "POST"}
    )
    assert bad.status_code == 400 and "access-control-allow-origin" not in bad.headers


def test_cors_origins_are_configurable_and_never_a_wildcard():
    assert parse_origins("*") == [] and parse_origins("http://a, *, ,http://b") == [
        "http://a",
        "http://b",
    ]
    c = make_env(api_cors_origins="*").client
    r = c.get("/health", headers={"origin": "https://anything.example"})
    assert "access-control-allow-origin" not in r.headers
    c = make_env(api_cors_origins="https://demo.example").client
    assert (
        c.get("/health", headers={"origin": "https://demo.example"}).headers[
            "access-control-allow-origin"
        ]
        == "https://demo.example"
    )


# ---- security headers ----------------------------------------------------------------------


def test_security_headers_on_every_kind_of_response():
    e = make_env(api_rate_limit=2)  # the invalid request below counts too
    responses = [
        e.client.get("/"),
        e.client.get("/health"),
        e.client.get("/static/app.js"),
        e.client.get("/nope"),  # 404
        e.client.get("/search", params={"q": ""}),  # 422
        e.client.get("/search", params={"q": "x"}),
        e.client.get("/search", params={"q": "x"}),  # 429
    ]
    e.search.error = RuntimeError("x")
    responses.append(e.client.get("/documents"))  # 500
    assert [r.status_code for r in responses] == [200, 200, 200, 404, 422, 200, 429, 500]
    for r in responses:
        assert r.headers["x-content-type-options"] == "nosniff", r.status_code
        assert r.headers["content-security-policy"] == SECURITY_HEADERS["Content-Security-Policy"]
        assert (
            r.headers["x-frame-options"] == "DENY" and r.headers["referrer-policy"] == "no-referrer"
        )


def test_csp_is_restrictive():
    csp = SECURITY_HEADERS["Content-Security-Policy"]
    assert "default-src 'none'" in csp and "script-src 'self'" in csp
    assert "unsafe-inline" not in csp and "unsafe-eval" not in csp and "*" not in csp
    assert "frame-ancestors 'none'" in csp


# ---- XSS -----------------------------------------------------------------------------------


def test_highlights_cannot_carry_markup_other_than_em():
    e = make_env()
    for hits in (
        e.client.get("/search", params={"q": "x"}).json()["hits"],
        e.client.get("/compare", params={"q": "x"}).json()["results"]["bm25"]["hits"],
    ):
        assert all("<script>" not in hl for h in hits for hl in h["highlights"])
    hit = e.client.get("/search", params={"q": "x"}).json()["hits"][0]
    assert hit["highlights"] == [
        "Die <em>Nationalbank</em> &lt;script&gt;alert(1)&lt;/script&gt; &amp; mehr"
    ]


def test_hostile_chunk_text_stays_plain_text_data():
    hit = HitOut.from_hit(SearchHit(CHUNKS[2], 1.0, 1, "bm25", [f"x {XSS}"]))
    assert "<script>" not in hit.highlights[0]
    assert (
        XSS in hit.text and XSS in hit.title
    )  # raw in JSON; the UI only uses textContent for these


def test_app_js_never_uses_html_sinks():
    js = (ROOT / "web" / "static" / "app.js").read_text()
    for sink in ("outerHTML", "insertAdjacentHTML", "document.write", "eval(", "new Function"):
        assert sink not in js, sink
    assert js.count("innerHTML") == 1  # only the textarea entity decoder
    assert "decoder.innerHTML" in js


def test_templates_use_no_safe_filter_and_no_inline_script_or_style():
    for path in (ROOT / "web" / "templates").glob("*.html"):
        text = path.read_text()
        assert "|safe" not in text and "| safe" not in text and "autoescape false" not in text
        assert "<style" not in text and " style=" not in text and "onclick=" not in text


# ---- error pages ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("path", "status", "needle"),
    [("/nope", 404, "Seite nicht gefunden"), ("/search?q=", 422, "darf nicht leer sein")],
)
def test_browsers_get_html_error_pages_and_apis_get_json(path, status, needle):
    c = make_env().client
    page = c.get(path, headers=HTML)
    assert page.status_code == status and page.headers["content-type"].startswith("text/html")
    assert needle in page.text and f"Fehler {status}" in page.text and "Zur Suche" in page.text
    api = c.get(path, headers={"accept": "application/json"})
    assert api.status_code == status and api.json()["message"]
    assert c.get(path).json()["error"]  # curl default */*


def test_html_429_and_500_pages():
    e = make_env(api_rate_limit=1)
    e.client.get("/search", params={"q": "x"})
    r = e.client.get("/search", params={"q": "x"}, headers=HTML)
    assert r.status_code == 429 and "Zu viele Anfragen" in r.text and "retry-after" in r.headers
    e.search.error = RuntimeError("secret")
    r = e.client.get("/stats", headers=HTML)
    assert r.status_code == 500 and "Interner Fehler" in r.text and "secret" not in r.text


def test_error_page_escapes_message():
    app = create_app(settings(), query_logger=make_env().logger)  # type: ignore[arg-type]

    @app.get("/boom")
    def boom():
        raise ApiError(418, "teapot", f"x {XSS}", hint=XSS)

    r = TestClient(app).get("/boom", headers=HTML)
    assert r.status_code == 418 and XSS not in r.text and "&lt;script&gt;" in r.text


# ---- query-log hand-off ---------------------------------------------------------------------


def logged(e):
    return [r for r in e.logger.records if r["path"] not in ("/health",)]


def test_search_sets_request_state_obs_for_the_query_log():
    e = make_env()
    e.client.get("/search", params={"q": "Leitzins Inflation", "method": "knn"})
    (rec,) = logged(e)
    assert (rec["path"], rec["http_method"], rec["status"]) == ("/search", "GET", 200)
    obs = rec["obs"]
    assert (obs["endpoint"], obs["method"], obs["query"], obs["n_hits"]) == (
        "search",
        "knn",
        "Leitzins Inflation",
        3,
    )
    assert obs["timings_ms"]["embed"] == 3.5 and obs["timings_ms"]["knn"] == 2.5
    assert set(obs) == {"endpoint", "method", "query", "n_hits", "timings_ms", "lang", "found"}
    assert obs["lang"] == "de" and obs["found"] is None


def test_ask_and_compare_set_obs():
    e = make_env()
    e.client.post("/ask", json={"question": "Wie hoch ist der Leitzins im Juni 2024?"})
    e.client.get("/compare", params={"q": "Zinsen"})
    ask, compare = logged(e)
    assert ask["obs"]["endpoint"] == "ask" and ask["obs"]["found"] is True
    assert ask["obs"]["n_hits"] == 3 and "llm" in ask["obs"]["timings_ms"]
    assert compare["obs"]["endpoint"] == "compare" and compare["obs"]["n_hits"] == 3


def test_failed_search_is_still_labelled_for_the_log():
    e = make_env()
    e.search.error = EsConnectionError("down")
    e.client.get("/search", params={"q": "Zins", "method": "bm25"})
    (rec,) = logged(e)
    assert rec["status"] == 503 and rec["obs"]["endpoint"] == "search"
    assert rec["obs"]["query"] == "Zins" and "n_hits" not in rec["obs"]


def test_default_app_registers_the_real_query_log_middleware():
    assert any(m.cls is QueryLogMiddleware for m in app.user_middleware)
