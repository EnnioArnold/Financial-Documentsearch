"""Elasticsearch client factory."""

from __future__ import annotations

import logging
import time

from elasticsearch import Elasticsearch

from finsearch.config import Settings, get_settings

log = logging.getLogger(__name__)

REQUEST_TIMEOUT_S = 30
MAX_RETRIES = 3
WAIT_POLL_S = 1.0


def get_es(settings: Settings | None = None) -> Elasticsearch:
    """Build a client with timeouts and retries (no network call happens here)."""
    s = settings or get_settings()
    return Elasticsearch(
        s.es_url, request_timeout=REQUEST_TIMEOUT_S, max_retries=MAX_RETRIES, retry_on_timeout=True
    )


def wait_for_es(es: Elasticsearch, timeout_s: float = 60.0) -> None:
    """Block until the cluster answers a health check at least 'yellow' (single node => yellow is fine)."""
    deadline = time.monotonic() + timeout_s
    last_error: Exception | None = None
    while True:
        try:
            es.cluster.health(wait_for_status="yellow", timeout="5s")
            return
        except Exception as exc:  # noqa: BLE001 - any connection problem means "not ready yet"
            last_error = exc
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Elasticsearch not ready after {timeout_s:.0f}s: {last_error}"
                ) from exc
            log.info("waiting for Elasticsearch: %s", exc)
            time.sleep(WAIT_POLL_S)
