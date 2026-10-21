"""Query log: one document per API request in the `finsearch-queries` index (feeds Kibana).

Privacy: the raw query text is only stored when `log_raw_queries` is true; otherwise only a short
hash (for "top queries" counts) and the length. No headers, bodies or keys are ever logged.
Writes happen on a single background thread so a slow or dead Elasticsearch never blocks a request.
"""

from __future__ import annotations

import hashlib
import json
import logging
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from typing import Any

from elasticsearch import Elasticsearch

from finsearch.config import Settings

log = logging.getLogger(__name__)
request_log = logging.getLogger("finsearch.request")

QUERY_HASH_LEN = 16
QUERY_KEYWORD_MAX_LEN = 512
TIMING_KEYS = ("embed", "bm25", "knn", "fusion", "rerank", "llm")
ES_TIMEOUT_S = 5

QUERIES_MAPPING: dict[str, Any] = {
    # dynamic=false: unexpected extra fields stay in _source but never become mapped/searchable
    "dynamic": "false",
    "properties": {
        "@timestamp": {"type": "date"},
        "path": {"type": "keyword"},
        "http_method": {"type": "keyword"},
        "status": {"type": "integer"},
        "error": {"type": "boolean"},
        "error_type": {"type": "keyword"},
        "endpoint": {"type": "keyword"},
        "method": {"type": "keyword"},
        "query_hash": {"type": "keyword"},
        "query": {"type": "keyword", "ignore_above": QUERY_KEYWORD_MAX_LEN},
        "query_len": {"type": "integer"},
        "n_hits": {"type": "integer"},
        "zero_hits": {"type": "boolean"},
        "total_ms": {"type": "float"},
        **{f"{key}_ms": {"type": "float"} for key in TIMING_KEYS},
        "lang": {"type": "keyword"},
        "found": {"type": "boolean"},
        "synthetic": {"type": "boolean"},  # only set by observability/seed_demo_queries.py
    },
}


def query_hash(query: str) -> str:
    """Stable id of a query: lowercase + collapsed whitespace, sha256, first 16 hex chars."""
    normalized = " ".join(query.lower().split())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:QUERY_HASH_LEN]


def build_doc(
    *,
    path: str,
    http_method: str,
    status: int,
    total_ms: float,
    error_type: str | None = None,
    obs: Mapping[str, Any] | None = None,
    log_raw_queries: bool = False,
) -> dict[str, Any]:
    """Assemble the log document. `obs` is what a route handler put into `request.state.obs`."""
    obs = obs or {}
    doc: dict[str, Any] = {
        "@timestamp": datetime.now(UTC).isoformat(),
        "path": path,
        "http_method": http_method,
        "status": status,
        "error": status >= 500 or error_type is not None,  # 4xx = client error, not ours
        "total_ms": round(total_ms, 2),
    }
    if error_type:
        doc["error_type"] = error_type
    for key in ("endpoint", "method", "lang", "found"):
        if obs.get(key) is not None:
            doc[key] = obs[key]
    query = obs.get("query")
    if isinstance(query, str):
        doc["query_hash"] = query_hash(query)
        doc["query_len"] = len(query)
        if log_raw_queries:
            doc["query"] = query
    n_hits = obs.get("n_hits")
    if isinstance(n_hits, int):
        doc["n_hits"] = n_hits
        doc["zero_hits"] = n_hits == 0
    timings = obs.get("timings_ms") or {}
    for key in TIMING_KEYS:
        if key in timings:
            doc[f"{key}_ms"] = round(float(timings[key]), 2)
    return doc


class QueryLogger:
    def __init__(self, es: Elasticsearch, index: str, log_raw_queries: bool = False) -> None:
        self._es = es
        self.index = index
        self.log_raw_queries = log_raw_queries
        self._pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="querylog")

    @classmethod
    def from_settings(cls, settings: Settings) -> QueryLogger:
        es = Elasticsearch(settings.es_url, request_timeout=ES_TIMEOUT_S, max_retries=0)
        return cls(es, settings.es_queries_index, settings.log_raw_queries)

    def ensure_index(self) -> bool:
        """Create the index with explicit mapping if missing. Idempotent; never raises."""
        try:
            if not self._es.indices.exists(index=self.index):
                self._es.indices.create(index=self.index, mappings=QUERIES_MAPPING)
            return True
        except Exception as exc:  # ES down at startup must not stop the API
            log.warning("could not ensure query-log index %s: %s", self.index, exc)
            return False

    def record(
        self,
        *,
        path: str,
        http_method: str,
        status: int,
        total_ms: float,
        error_type: str | None = None,
        obs: Mapping[str, Any] | None = None,
    ) -> None:
        """Log one request as a JSON line and queue the ES write (returns immediately)."""
        doc = build_doc(
            path=path,
            http_method=http_method,
            status=status,
            total_ms=total_ms,
            error_type=error_type,
            obs=obs,
            log_raw_queries=self.log_raw_queries,
        )
        request_log.info(json.dumps(doc, ensure_ascii=False))
        self._pool.submit(self._write, doc)

    def _write(self, doc: dict[str, Any]) -> None:
        try:
            self._es.index(index=self.index, document=doc)
        except Exception as exc:  # drop the document, never fail anything
            log.warning("query-log write dropped: %s", exc)

    def flush(self) -> None:
        """Block until all queued writes are done (tests, shutdown)."""
        self._pool.submit(lambda: None).result()

    def close(self) -> None:
        self._pool.shutdown(wait=True)
