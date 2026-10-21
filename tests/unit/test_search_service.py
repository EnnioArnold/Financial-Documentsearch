"""SearchService logic with a fake Elasticsearch client (no network, no models)."""

from typing import Any

import pytest
from elasticsearch import NotFoundError
from fakes import FakeReranker, HashEmbedder, make_chunk

from finsearch.config import Settings
from finsearch.contracts import SearchFilters
from finsearch.search.mapping import chunk_to_doc
from finsearch.search.service import SearchService


def _hit(chunk_id: str, text: str, score: float, **extra: Any) -> dict[str, Any]:
    doc = chunk_to_doc(make_chunk(chunk_id, text), [])
    for dropped in ("embedding", "text_de", "text_en"):
        doc.pop(dropped)
    return {"_source": doc, "_score": score, **extra}


class FakeES:
    """Answers BM25 (has `query`) and kNN (has `knn`) calls with canned hits and records the calls."""

    def __init__(self, bm25_hits: list[dict[str, Any]], knn_hits: list[dict[str, Any]]) -> None:
        self.bm25_hits, self.knn_hits = bm25_hits, knn_hits
        self.calls: list[dict[str, Any]] = []

    def search(self, index: str, **kw: Any) -> dict[str, Any]:
        self.calls.append({"index": index, **kw})
        hits = self.knn_hits if "knn" in kw else self.bm25_hits
        return {"hits": {"hits": hits}}


BM25 = [
    _hit("a-0000", "alpha text one", 9.0, highlight={"text": ["<em>alpha</em> text"]}),
    _hit("b-0000", "beta text two", 5.0),
]
KNN = [_hit("c-0000", "gamma text three", 0.9), _hit("a-0000", "alpha text one", 0.8)]


def make_service(reranker: Any = None, **kw: Any) -> tuple[SearchService, FakeES]:
    es = FakeES(BM25, KNN)
    return SearchService(es, HashEmbedder(), reranker, Settings(), **kw), es  # type: ignore[arg-type]


def test_bm25_result_and_timings():
    svc, es = make_service()
    res = svc.search("alpha", "bm25", k=5)
    assert [h.chunk.chunk_id for h in res.hits] == ["a-0000", "b-0000"]
    assert [h.rank for h in res.hits] == [1, 2]
    assert res.hits[0].highlights == ["<em>alpha</em> text"]
    assert res.hits[1].highlights == ["beta text two"]  # no ES highlight -> plain snippet
    assert set(res.timings_ms) == {"bm25"} and res.total_ms >= 0
    assert es.calls[0]["index"] == "finsearch" and es.calls[0]["size"] == 5


def test_knn_embeds_and_reports_timings():
    svc, es = make_service()
    res = svc.search("gamma", "knn", k=2)
    assert [h.method for h in res.hits] == ["knn", "knn"]
    assert set(res.timings_ms) == {"embed", "knn"}
    assert len(es.calls[0]["knn"]["query_vector"]) == 384


def test_hybrid_fuses_with_rrf_and_keeps_bm25_highlight():
    svc, es = make_service()
    res = svc.search("alpha", "hybrid", k=3, filters=SearchFilters(lang="en"))
    # a-0000: bm25 rank 1 + knn rank 2 = 1/61 + 1/62 ; c-0000: knn rank 1 = 1/61 ; b-0000: bm25 rank 2 = 1/62
    assert [h.chunk.chunk_id for h in res.hits] == ["a-0000", "c-0000", "b-0000"]
    assert res.hits[0].score == pytest.approx(1 / 61 + 1 / 62)
    assert res.hits[0].highlights == ["<em>alpha</em> text"]
    assert res.hits[1].highlights == ["gamma text three"]
    assert set(res.timings_ms) == {"embed", "bm25", "knn", "fusion"}
    # both retrievers got the same filter and the candidate pool size
    assert all(c.get("size") == 50 for c in es.calls)
    assert es.calls[0]["query"]["bool"]["filter"] == es.calls[1]["knn"]["filter"]["bool"]["filter"]


def test_hybrid_rerank_uses_reranker_and_adds_timing():
    svc, _ = make_service(FakeReranker())
    res = svc.search("gamma three", "hybrid_rerank", k=2)
    assert res.hits[0].chunk.chunk_id == "c-0000"
    assert {h.method for h in res.hits} == {"hybrid_rerank"}
    assert len(res.hits) == 2 and set(res.timings_ms) == {
        "embed",
        "bm25",
        "knn",
        "fusion",
        "rerank",
    }


def test_hybrid_rerank_without_reranker_is_a_clear_error():
    svc, _ = make_service()
    with pytest.raises(ValueError, match="reranker"):
        svc.search("x", "hybrid_rerank")


@pytest.mark.parametrize(("method", "k"), [("semantic", 10), ("bm25", 0), ("bm25", 101)])
def test_bad_arguments_raise(method, k):
    svc, _ = make_service()
    with pytest.raises(ValueError):
        svc.search("x", method, k)


@pytest.mark.parametrize("method", ["bm25", "knn", "hybrid", "hybrid_rerank"])
@pytest.mark.parametrize("query", ["", "   ", "\n\t"])
def test_blank_query_returns_empty_without_touching_es(method, query):
    svc, es = make_service(FakeReranker())
    res = svc.search(query, method)
    assert res.hits == [] and res.method == method and res.query == query
    assert res.timings_ms == {} and es.calls == []


def test_very_long_query_is_truncated_before_embedding():
    svc, es = make_service()
    svc.search("wort " * 10_000, "bm25")
    assert len(es.calls[0]["query"]["bool"]["must"][0]["multi_match"]["query"]) <= 1000


def test_stats_on_missing_index_is_zero():
    class Missing:
        def search(self, **_: Any) -> None:
            raise NotFoundError("nope", meta=None, body={})  # type: ignore[arg-type]

    svc = SearchService(Missing(), HashEmbedder(), None, Settings())  # type: ignore[arg-type]
    assert svc.stats() == {"documents": 0, "chunks": 0, "by_source": {}, "by_lang": {}}


def test_get_chunks_empty_list_skips_es():
    svc, es = make_service()
    assert svc.get_chunks([]) == [] and es.calls == []
