"""Fixtures for tests against the running Elasticsearch: one throwaway index with the synthetic corpus."""

from __future__ import annotations

import uuid
from collections.abc import Iterator

import pytest
from elasticsearch import Elasticsearch
from fakes import HashEmbedder
from synthetic_corpus import CORPUS

from finsearch.config import Settings
from finsearch.search.indexer import index_chunks
from finsearch.search.service import SearchService


@pytest.fixture(scope="module")
def corpus_index(es: Elasticsearch, test_settings: Settings) -> Iterator[str]:
    name = f"{test_settings.es_index_alias}-{uuid.uuid4().hex[:8]}"
    stats = index_chunks(CORPUS, test_settings, HashEmbedder(), index_name=name, es=es)
    assert stats["failed"] == 0
    yield name
    es.indices.delete(index=name, ignore_unavailable=True)


@pytest.fixture(scope="module")
def service(es: Elasticsearch, corpus_index: str, test_settings: Settings) -> SearchService:
    from fakes import FakeReranker

    return SearchService(es, HashEmbedder(), FakeReranker(), test_settings, index=corpus_index)
