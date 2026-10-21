"""Shared fixtures. Only Elasticsearch-related ones live here (integration tests)."""

from __future__ import annotations

import logging
import uuid
from collections.abc import Iterator

import pytest
from elasticsearch import Elasticsearch

from finsearch.config import Settings
from finsearch.search.es_client import get_es, wait_for_es

ES_WAIT_S = 30.0


@pytest.fixture(scope="session")
def test_settings() -> Settings:
    """Settings for tests: default ES URL, but an alias name that can never collide with real data."""
    return Settings(es_index_alias=f"finsearch-test-{uuid.uuid4().hex[:8]}")


@pytest.fixture(scope="session")
def es(test_settings: Settings) -> Iterator[Elasticsearch]:
    """Client for the running dev Elasticsearch (fails loudly if it is not up)."""
    client = get_es(test_settings)
    wait_for_es(client, ES_WAIT_S)
    yield client
    # drop everything this run created (indices and alias share the unique prefix)
    prefix = test_settings.es_index_alias
    try:  # (wildcard deletes are disabled by default in ES 9, so list names first)
        for name in client.indices.get(index=f"{prefix}*", ignore_unavailable=True):
            client.indices.delete(index=name, ignore_unavailable=True)
    except Exception:  # noqa: BLE001 - cleanup must never fail the run
        logging.getLogger(__name__).warning("cleanup of %s* failed", prefix)
    client.close()


@pytest.fixture
def index_name(es: Elasticsearch, test_settings: Settings) -> Iterator[str]:
    """A fresh unique index name per test (not created); deleted afterwards if the test created it."""
    name = f"{test_settings.es_index_alias}-{uuid.uuid4().hex[:8]}"
    yield name
    es.indices.delete(index=name, ignore_unavailable=True)
