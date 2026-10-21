"""Embedder / reranker wrappers with injected fake models (the real ones are covered by integration tests)."""

import numpy as np
import pytest
from fakes import make_chunk

from finsearch.contracts import SearchHit
from finsearch.search.embedder import (
    MODEL_REVISIONS,
    PASSAGE_PREFIX,
    QUERY_PREFIX,
    SentenceTransformerEmbedder,
    get_embedder,
)
from finsearch.search.es_client import get_es, wait_for_es
from finsearch.search.rerank import CrossEncoderReranker


class FakeST:
    def __init__(self) -> None:
        self.seen: list[str] = []

    def encode(self, texts, **kw):
        self.seen += texts
        assert kw["normalize_embeddings"] is True
        return np.ones((len(texts), 4), dtype=np.float32) / 2


def test_embedder_adds_prefixes_and_returns_python_floats():
    emb = SentenceTransformerEmbedder("m", dim=4)
    emb._model = fake = FakeST()
    q = emb.embed_queries(["Zins"])
    p = emb.embed_passages(["Zins", "Bilanz"])
    assert fake.seen == [QUERY_PREFIX + "Zins", PASSAGE_PREFIX + "Zins", PASSAGE_PREFIX + "Bilanz"]
    assert q == [[0.5] * 4] and len(p) == 2 and type(q[0][0]) is float
    assert emb.embed_queries([]) == []


def test_get_embedder_uses_settings_and_default_model_is_pinned():
    emb = get_embedder()
    assert emb.dim == 384
    assert emb._model is None  # type: ignore[attr-defined]  # lazy: nothing loaded yet
    assert len(MODEL_REVISIONS["intfloat/multilingual-e5-small"]) == 40


class FakeCE:
    def predict(self, pairs, **_):
        return np.array([float(len(t.split())) for _, t in pairs])  # longer text = better


def _hits():
    return [
        SearchHit(make_chunk(f"d-{i:04d}", "w " * n), 0.1, i + 1, "hybrid", ["hl"])
        for i, n in enumerate([1, 3, 2])
    ]


def test_reranker_sorts_by_model_score_and_truncates():
    rr = CrossEncoderReranker("fake")
    rr._model = FakeCE()
    out = rr.rerank("q", _hits(), top_k=2)
    assert [h.chunk.chunk_id for h in out] == ["d-0001", "d-0002"]
    assert [h.rank for h in out] == [1, 2] and out[0].score == 3.0
    assert {h.method for h in out} == {"hybrid_rerank"} and out[0].highlights == ["hl"]
    assert rr.rerank("q", [], top_k=3) == []


def test_reranker_is_lazy():
    assert CrossEncoderReranker("fake")._model is None


def test_wait_for_es_times_out_with_clear_message():
    class Down:
        class cluster:  # noqa: N801
            @staticmethod
            def health(**_):
                raise ConnectionError("refused")

    with pytest.raises(TimeoutError, match="not ready"):
        wait_for_es(Down(), timeout_s=0.0)  # type: ignore[arg-type]


def test_get_es_builds_client_without_network():
    assert get_es() is not None
