"""Index mapping, versioned index creation and alias switching (zero-downtime re-index)."""

from __future__ import annotations

import logging
from datetime import date
from typing import Any

from elasticsearch import Elasticsearch, NotFoundError

from finsearch.config import Settings, get_settings
from finsearch.contracts import Chunk

log = logging.getLogger(__name__)

# HNSW: m = graph connectivity, ef_construction = build-time beam width. Values are the ES defaults,
# made explicit on purpose: since ES 9 the default for >=384 dims is quantised "bbq_hnsw", and we want
# full-precision vectors (the corpus is small, recall matters more than memory).
HNSW_M = 16
HNSW_EF_CONSTRUCTION = 100
TITLE_KEYWORD_IGNORE_ABOVE = 512


def index_body(dim: int) -> dict[str, Any]:
    """Settings + mappings for one chunk index. One chunk text is analysed three ways (ADR-003)."""
    return {
        "settings": {"number_of_shards": 1, "number_of_replicas": 0},
        "mappings": {
            "dynamic": "strict",
            "properties": {
                "chunk_id": {"type": "keyword"},
                "doc_id": {"type": "keyword"},
                "source": {"type": "keyword"},
                "doc_type": {"type": "keyword"},
                "lang": {"type": "keyword"},
                "date": {
                    "type": "date",
                    "format": "strict_date_optional_time",
                    "ignore_malformed": False,
                },
                "page": {"type": "integer"},
                "ord": {"type": "integer"},
                "title": {
                    "type": "text",
                    "fields": {
                        "keyword": {"type": "keyword", "ignore_above": TITLE_KEYWORD_IGNORE_ABOVE}
                    },
                },
                "url": {"type": "keyword"},
                "text": {"type": "text", "analyzer": "standard"},
                "text_de": {"type": "text", "analyzer": "german"},
                "text_en": {"type": "text", "analyzer": "english"},
                "embedding": {
                    "type": "dense_vector",
                    "dims": dim,
                    "index": True,
                    "similarity": "cosine",
                    "index_options": {
                        "type": "hnsw",
                        "m": HNSW_M,
                        "ef_construction": HNSW_EF_CONSTRUCTION,
                    },
                },
            },
        },
    }


def chunk_to_doc(chunk: Chunk, embedding: list[float]) -> dict[str, Any]:
    """Chunk -> ES document. text_de/text_en get the same text, only the analyzer differs."""
    return {
        "chunk_id": chunk.chunk_id,
        "doc_id": chunk.doc_id,
        "source": chunk.source,
        "doc_type": chunk.doc_type,
        "lang": chunk.lang,
        "date": chunk.date.isoformat() if chunk.date else None,
        "page": chunk.page,
        "ord": chunk.ord,
        "title": chunk.title,
        "url": chunk.url,
        "text": chunk.text,
        "text_de": chunk.text,
        "text_en": chunk.text,
        "embedding": embedding,
    }


def doc_to_chunk(src: dict[str, Any]) -> Chunk:
    """ES _source -> Chunk (inverse of chunk_to_doc, minus the embedding)."""
    raw_date = src.get("date")
    return Chunk(
        chunk_id=src["chunk_id"],
        doc_id=src["doc_id"],
        ord=src["ord"],
        text=src["text"],
        page=src["page"],
        source=src["source"],
        title=src.get("title", ""),
        url=src.get("url", ""),
        lang=src["lang"],
        doc_type=src["doc_type"],
        date=date.fromisoformat(raw_date[:10]) if raw_date else None,
    )


def create_index(
    es: Elasticsearch,
    settings: Settings | None = None,
    version: str | None = None,
    index_name: str | None = None,
) -> str:
    """Create `<alias>-<version>` (or `index_name`) if missing. Idempotent. Returns the index name."""
    s = settings or get_settings()
    name = index_name or (f"{s.es_index_alias}-{version}" if version else s.index_name)
    if es.indices.exists(index=name):
        log.debug("index %s already exists", name)
        return name
    es.indices.create(index=name, **index_body(s.embedding_dim))
    log.info("created index %s", name)
    return name


def switch_alias(es: Elasticsearch, alias: str, index_name: str) -> None:
    """Atomically point `alias` at `index_name` (removes it from every other index in the same call)."""
    actions: list[dict[str, Any]] = []
    try:
        current = es.indices.get_alias(name=alias)
        actions += [
            {"remove": {"index": idx, "alias": alias}} for idx in current if idx != index_name
        ]
    except NotFoundError:
        pass
    actions.append({"add": {"index": index_name, "alias": alias}})
    es.indices.update_aliases(actions=actions)
    log.info("alias %s -> %s", alias, index_name)


def ensure_index(
    es: Elasticsearch, settings: Settings | None = None, index_name: str | None = None
) -> str:
    """Create the index if needed and, if the alias does not exist yet, point it there."""
    s = settings or get_settings()
    name = create_index(es, s, index_name=index_name)
    if index_name is None and not es.indices.exists_alias(name=s.es_index_alias):
        switch_alias(es, s.es_index_alias, name)
    return name
