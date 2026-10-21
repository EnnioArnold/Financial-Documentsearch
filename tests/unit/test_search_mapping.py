"""Mapping dict sanity and chunk <-> document conversion (no Elasticsearch)."""

from datetime import date

import pytest
from fakes import make_chunk

from finsearch.contracts import Chunk
from finsearch.search.mapping import (
    HNSW_EF_CONSTRUCTION,
    HNSW_M,
    chunk_to_doc,
    doc_to_chunk,
    index_body,
)

REQUIRED_FIELDS = {
    "chunk_id",
    "doc_id",
    "source",
    "doc_type",
    "lang",
    "date",
    "page",
    "ord",
    "title",
    "url",
    "text",
    "text_de",
    "text_en",
    "embedding",
}


def test_mapping_fields_and_analyzers():
    props = index_body(384)["mappings"]["properties"]
    assert set(props) == REQUIRED_FIELDS
    assert props["text_de"]["analyzer"] == "german"
    assert props["text_en"]["analyzer"] == "english"
    assert props["text"]["analyzer"] == "standard"
    assert props["title"]["fields"]["keyword"]["type"] == "keyword"
    assert props["date"]["ignore_malformed"] is False
    for kw in ("chunk_id", "doc_id", "source", "doc_type", "lang", "url"):
        assert props[kw]["type"] == "keyword"


def test_mapping_vector_settings_follow_dim():
    emb = index_body(1024)["mappings"]["properties"]["embedding"]
    assert emb["dims"] == 1024 and emb["similarity"] == "cosine" and emb["index"] is True
    assert emb["index_options"] == {
        "type": "hnsw",
        "m": HNSW_M,
        "ef_construction": HNSW_EF_CONSTRUCTION,
    }


@pytest.mark.parametrize("d", [None, date(2024, 3, 21)])
def test_chunk_roundtrip(d):
    chunk = make_chunk("doc-0003", "Text mit Zins", ord=3, page=7, lang="de", date=d)
    doc = chunk_to_doc(chunk, [0.0] * 384)
    assert doc["text"] == doc["text_de"] == doc["text_en"] == "Text mit Zins"
    assert doc["date"] == (d.isoformat() if d else None)
    assert set(doc) == REQUIRED_FIELDS
    without_vectors = {k: v for k, v in doc.items() if k not in ("text_de", "text_en", "embedding")}
    restored: Chunk = doc_to_chunk(without_vectors)
    assert restored == chunk
