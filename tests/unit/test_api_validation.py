"""Input validation: bad input is a clear 422, never a stack trace."""

from __future__ import annotations

import pytest
from api_fakes import make_env

from finsearch.api.schemas import MAX_QUERY_CHARS, safe_highlight


@pytest.fixture(scope="module")
def client():
    return make_env().client


def assert_422(r, *fragments):
    assert r.status_code == 422, r.text
    body = r.json()
    assert body["error"] == "validation_error" and body["detail"]
    assert all(f in body["message"] for f in fragments), body["message"]
    assert "Traceback" not in r.text


@pytest.mark.parametrize("q", ["", "   ", "\t\n"])
def test_blank_query_is_422_with_clear_message(client, q):
    assert_422(client.get("/search", params={"q": q}), "q", "darf nicht leer sein")
    assert_422(client.get("/compare", params={"q": q}), "q", "darf nicht leer sein")
    assert_422(client.post("/ask", json={"question": q}), "question", "darf nicht leer sein")


def test_missing_query_is_422(client):
    assert_422(client.get("/search"), "q")
    assert_422(client.post("/ask", json={}), "question")


def test_query_length_limit_applies_after_strip(client):
    assert client.get("/search", params={"q": " " + "a" * MAX_QUERY_CHARS + " "}).status_code == 200
    assert_422(client.get("/search", params={"q": "a" * (MAX_QUERY_CHARS + 1)}), "q")
    assert_422(client.post("/ask", json={"question": "a" * (MAX_QUERY_CHARS + 1)}), "question")


@pytest.mark.parametrize("k", ["0", "51", "-1", "abc", "1.5"])
def test_k_out_of_range(client, k):
    assert_422(client.get("/search", params={"q": "x", "k": k}), "k")
    assert_422(client.get("/compare", params={"q": "x", "k": k}), "k")


@pytest.mark.parametrize("k", [1, 50])
def test_k_bounds_accepted(client, k):
    assert client.get("/search", params={"q": "x", "k": k}).status_code == 200


def test_ask_k_range(client):
    assert_422(client.post("/ask", json={"question": "x", "k": 0}), "k")
    assert_422(client.post("/ask", json={"question": "x", "k": 51}), "k")


def test_unknown_method(client):
    assert_422(client.get("/search", params={"q": "x", "method": "magic"}), "method")
    assert_422(client.post("/ask", json={"question": "x", "method": "magic"}), "method")
    assert_422(client.get("/search", params={"q": "x", "method": "BM25"}), "method")


@pytest.mark.parametrize("bad", ["2024-13-01", "yesterday", "2024-02-30", "01.02.2024"])
def test_dates_must_be_iso(client, bad):
    assert_422(client.get("/search", params={"q": "x", "from": bad}), "from")
    assert_422(client.get("/search", params={"q": "x", "to": bad}), "to")
    assert_422(client.post("/ask", json={"question": "x", "filters": {"from": bad}}), "from")


def test_from_must_not_be_after_to(client):
    p = {"q": "x", "from": "2024-06-01", "to": "2024-01-01"}
    assert_422(client.get("/search", params=p), "'from' darf nicht nach 'to' liegen")
    body = {"question": "x", "filters": {"from": "2024-06-01", "to": "2024-01-01"}}
    assert_422(client.post("/ask", json=body), "'from' darf nicht nach 'to' liegen")
    assert (
        client.get("/search", params={**p, "to": "2024-06-01"}).status_code == 200
    )  # equal is fine


@pytest.mark.parametrize("lang", ["fr", "other", "DE", "deutsch"])
def test_lang_only_de_or_en(client, lang):
    assert_422(client.get("/search", params={"q": "x", "lang": lang}), "lang")


@pytest.mark.parametrize("source", ["Snb", "a b", "../etc", "snb;drop", "a" * 41, "snb,fed"])
def test_source_pattern(client, source):
    assert_422(client.get("/search", params={"q": "x", "source": source}), "source")
    assert_422(client.get("/documents", params={"source": source}), "source")
    assert_422(client.post("/ask", json={"question": "x", "filters": {"source": source}}), "source")


@pytest.mark.parametrize("source", ["snb", "fed", "other", "my_src-2"])
def test_valid_sources(client, source):
    assert client.get("/search", params={"q": "x", "source": source}).status_code == 200


def test_doc_type_pattern(client):
    assert_422(client.get("/search", params={"q": "x", "doc_type": "Speech!"}), "doc_type")


def test_ask_rejects_unknown_fields(client):
    assert_422(client.post("/ask", json={"question": "x", "foo": 1}), "foo")
    assert_422(client.post("/ask", json={"question": "x", "filters": {"foo": 1}}), "foo")


def test_ask_rejects_non_json_body(client):
    r = client.post("/ask", content="question=x", headers={"content-type": "text/plain"})
    assert r.status_code == 422


@pytest.mark.parametrize("params", [{"limit": 0}, {"limit": 101}, {"offset": -1}, {"limit": "x"}])
def test_documents_paging_bounds(client, params):
    assert_422(client.get("/documents", params=params))


def test_unknown_route_and_method_are_json_errors(client):
    r = client.get("/nope")
    assert r.status_code == 404 and r.json()["error"] == "http_error"
    r = client.post("/search", json={})
    assert r.status_code == 405 and r.json()["error"] == "http_error"


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("plain <em>hit</em> text", "plain <em>hit</em> text"),
        ("a &amp; b &lt;b&gt;", "a &amp; b &lt;b&gt;"),  # already escaped: unchanged (idempotent)
        ("<script>alert(1)</script>", "&lt;script&gt;alert(1)&lt;/script&gt;"),
        ("<em onclick=x>hit</em>", "&lt;em onclick=x&gt;hit</em>"),
        ("<img src=x onerror=y> <em>ok</em>", "&lt;img src=x onerror=y&gt; <em>ok</em>"),
        ("quote \" and ' stay safe", "quote &quot; and &#x27; stay safe"),
        ("", ""),
    ],
)
def test_safe_highlight(raw, expected):
    assert safe_highlight(raw) == expected
    assert safe_highlight(safe_highlight(raw)) == expected
