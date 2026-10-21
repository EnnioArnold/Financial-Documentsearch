"""Semantic behaviour with the REAL multilingual-e5-small model (downloads ~470 MB on first run)."""

import uuid

import pytest
from synthetic_corpus import CORPUS, JUDGED_QUERIES

from finsearch.contracts import SearchHit
from finsearch.search.embedder import get_embedder
from finsearch.search.indexer import index_chunks
from finsearch.search.rerank import get_reranker
from finsearch.search.service import SearchService

pytestmark = pytest.mark.integration


@pytest.fixture(scope="module")
def real_service(es, test_settings):
    embedder = get_embedder(test_settings)
    try:
        embedder.embed_queries(["probe"])
    except Exception as exc:  # noqa: BLE001 - offline / blocked download => skip, do not fail
        pytest.skip(f"embedding model not available: {exc}")
    name = f"{test_settings.es_index_alias}-real-{uuid.uuid4().hex[:6]}"
    index_chunks(CORPUS, test_settings, embedder, index_name=name, es=es)
    yield SearchService(es, embedder, None, test_settings, index=name)
    es.indices.delete(index=name, ignore_unavailable=True)


def test_knn_finds_german_paraphrase_without_shared_keywords(real_service):
    # "Teuerung" never appears in the query; BM25 cannot know it means inflation
    q = "Wie stark steigen die Preise in der Schweiz"
    got = [h.chunk.chunk_id for h in real_service.search(q, "knn", k=3).hits]
    assert "snb-speech-0001" in got


def test_knn_paraphrase_same_language(real_service):
    q = "what happens to household consumption when borrowing costs go up"
    assert real_service.search(q, "knn", k=1).hits[0].chunk.chunk_id == "fed-speech-0001"


def _scores(service, method: str, top: int = 3) -> tuple[float, float]:
    """(MRR@10, recall@top) over the judged queries."""
    rr = hit = 0.0
    for query, relevant in JUDGED_QUERIES:
        got = [h.chunk.chunk_id for h in service.search(query, method, k=10).hits]
        rr += 1 / (got.index(relevant) + 1) if relevant in got else 0.0
        hit += relevant in got[:top]
    return rr / len(JUDGED_QUERIES), hit / len(JUDGED_QUERIES)


def test_hybrid_recall_is_at_least_as_good_as_each_single_method(real_service):
    # RRF cannot guarantee a better *first* rank (see MRR in the output), but whatever one method puts
    # in the top 3 should still make it into the fused top 3 on this small example.
    res = {m: _scores(real_service, m) for m in ("bm25", "knn", "hybrid")}
    print("(MRR@10, recall@3):", {m: tuple(round(x, 3) for x in v) for m, v in res.items()})
    assert res["hybrid"][1] >= max(res["bm25"][1], res["knn"][1])


def test_hybrid_rerank_with_real_cross_encoder(real_service, test_settings):
    reranker = get_reranker(test_settings)
    try:
        reranker.rerank("probe", [SearchHit(CORPUS[0], 0.0, 1, "hybrid")], 1)
    except Exception as exc:  # noqa: BLE001 - offline / blocked download => skip, do not fail
        pytest.skip(f"reranker model not available: {exc}")
    service = SearchService(
        real_service.es, real_service.embedder, reranker, test_settings, index=real_service.index
    )
    res = service.search(
        "What happens to household spending when interest rates rise?", "hybrid_rerank", k=3
    )
    assert res.hits[0].chunk.chunk_id == "fed-speech-0001"
    assert res.hits[0].method == "hybrid_rerank" and "rerank" in res.timings_ms
    assert res.hits[0].score >= res.hits[-1].score
